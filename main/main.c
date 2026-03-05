#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "FreeRTOS.h"

#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_continuous.h"
#include "esp_adc/adc_filter.h"
#include "esp_attr.h"
#include "esp_crc.h"
#include "esp_dsp.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "led_strip.h"
#include "string.h"
#include "tinyusb.h"
#include "tusb_cdc_acm.h"

#if CONFIG_ENABLE_I2S_WAVE_GEN || CONFIG_ENABLE_SDM_WAVE_GEN
#include "wave_gen.h"
#endif


static const char *TAG = "adc_fft";

#if defined(__GNUC__) && !defined(__clang__)
#define OPTIMIZE_O2 __attribute__((optimize("O2")))
#else
#define OPTIMIZE_O2
#endif

#define FFT_SIZE CONFIG_ADC_FFT_SIZE
#define ADC_SPS CONFIG_ADC_FS

/* Build invariants: transport builds must not modify spectra. */
#if CONFIG_FFT_BUILD_TRANSPORT
#if CONFIG_ADC_ENABLE_WINDOW || CONFIG_OUTPUT_TOPK_ENABLE ||                \
  (CONFIG_OUTPUT_CUTOFF_HZ > 0) || CONFIG_ADC_IIR_FILTER_ENABLE
#error "Transport build forbids windowing/top-K/cutoff/IIR filtering"
#endif
#endif

#define CHANNELS 2
#define SAMPLE_MAX 4096

#define LED_STRIP_RMT_RES_HZ (10 * 1000 * 1000)

#define SPIKE_DB_THRESHOLD 20.0f
#define SPIKE_RATIO_THRESHOLD 8.0f
#define SPIKE_MIN_AVG_LINEAR 1e-7f

DRAM_ATTR __attribute__((aligned(16))) static float ch0[FFT_SIZE];
DRAM_ATTR __attribute__((aligned(16))) static float ch1[FFT_SIZE];
DRAM_ATTR __attribute__((aligned(16))) static float win[FFT_SIZE];

static adc_cali_handle_t cali_handle;

typedef struct __attribute__((packed)) {
  uint64_t t_ms;
  uint32_t sps;
  uint32_t fft;
  char label[4];
  float data[FFT_SIZE];
} frame_payload_t;

#define FRAME_MAGIC_0 0xF0   // "FOOD"
#define FRAME_MAGIC_1 0x0D   // "FOOD"
#define FRAME_MAGIC_2 0xF0   // "FOOD"
#define FRAME_MAGIC_3 0x0D   // "FOOD"
#define FRAME_VERSION 0x02   // Version 2: uint64_t timestamp
#define FRAME_HEADER_SIZE 9u // 4 magic + 1 version + 2 seq + 2 payload length
#define FRAME_PAYLOAD_SIZE (sizeof(frame_payload_t)) // Real data
#define FRAME_TOTAL_SIZE                                                       \
  (FRAME_HEADER_SIZE + FRAME_PAYLOAD_SIZE + 4u) // +4 for CRC32
_Static_assert(FRAME_PAYLOAD_SIZE <= UINT16_MAX,
               "payload length must fit in uint16");
_Static_assert((FRAME_PAYLOAD_SIZE % 4u) == 0u,
               "payload must stay 32-bit aligned");

typedef struct {
  uint8_t bytes[FRAME_TOTAL_SIZE];
} frame_wire_t;

static QueueHandle_t g_frameq;
static uint16_t g_seq;
static tinyusb_cdcacm_itf_t g_data_port = TINYUSB_CDC_ACM_0;
static TaskHandle_t g_cdc_task_handle;
static TaskHandle_t g_adc_task_handle;

#if CONFIG_STATUS_NEOPIXEL_ENABLE
static led_strip_handle_t g_led_strip;
static SemaphoreHandle_t g_led_lock;
#endif

// Task stack size: 4096 words (~16 KB). CDC TX uses small locals and no large
// stack arrays; 4K words leaves comfortable headroom for TinyUSB calls.
#define CDC_TX_TASK_STACK_SIZE 4096
// FFT producer task stack size: 8192 words (~32 KB). FFT math runs in-place on
// globals; stack usage is modest but we keep extra headroom for IDF/DSP calls.
#define ADC_TASK_STACK_SIZE 8192

static inline void write_u16_le(uint8_t *dst, uint16_t value) {
  dst[0] = (uint8_t)(value & 0xFFu);
  dst[1] = (uint8_t)((value >> 8) & 0xFFu);
}

static inline void write_u32_le(uint8_t *dst, uint32_t value) {
  dst[0] = (uint8_t)(value & 0xFFu);
  dst[1] = (uint8_t)((value >> 8) & 0xFFu);
  dst[2] = (uint8_t)((value >> 16) & 0xFFu);
  dst[3] = (uint8_t)((value >> 24) & 0xFFu);
}

static inline void write_u64_le(uint8_t *dst, uint64_t value) {
  write_u32_le(dst, (uint32_t)(value & 0xFFFFFFFFu));
  write_u32_le(dst + 4, (uint32_t)((value >> 32) & 0xFFFFFFFFu));
}

static uint32_t frame_crc32(const uint8_t *data, size_t len) {
  return esp_crc32_le(0, data, len);
}

static void build_frame(frame_wire_t *out, const float *data,
                        const char label[4], uint64_t t_ms, uint32_t sps,
                        uint16_t seq) {
  uint8_t *dst = out->bytes;
  dst[0] = FRAME_MAGIC_0;
  dst[1] = FRAME_MAGIC_1;
  dst[2] = FRAME_MAGIC_2;
  dst[3] = FRAME_MAGIC_3;
  dst[4] = FRAME_VERSION;
  write_u16_le(dst + 5, seq);
  write_u16_le(dst + 7, (uint16_t)FRAME_PAYLOAD_SIZE);

  size_t off = FRAME_HEADER_SIZE;
  write_u64_le(dst + off, t_ms);
  off += sizeof(uint64_t);
  write_u32_le(dst + off, sps);
  off += sizeof(uint32_t);
  write_u32_le(dst + off, FFT_SIZE);
  off += sizeof(uint32_t);
  memcpy(dst + off, label, 4);
  off += 4;
  memcpy(dst + off, data, sizeof(float) * FFT_SIZE);
  off += sizeof(float) * FFT_SIZE;

  uint32_t crc = frame_crc32(dst, FRAME_HEADER_SIZE + FRAME_PAYLOAD_SIZE);
  write_u32_le(dst + off, crc);
}

#if CONFIG_DIAG_CDC_ENABLE
typedef struct {
  uint64_t total_bytes_sent;    /* lifetime */
  uint32_t window_bytes;        /* bytes in current window */
  uint64_t window_start_ms;
  uint32_t wa_min;
  uint32_t wa_max;
  uint64_t wa_sum;
  uint32_t wa_count;
} diag_cdc_t;
static diag_cdc_t s_diag_cdc;

static void diag_cdc_reset_window(uint64_t now_ms)
{
  s_diag_cdc.window_bytes = 0;
  s_diag_cdc.window_start_ms = now_ms;
  s_diag_cdc.wa_min = UINT32_MAX;
  s_diag_cdc.wa_max = 0;
  s_diag_cdc.wa_sum = 0;
  s_diag_cdc.wa_count = 0;
}

static void diag_cdc_sample_wa(uint32_t v)
{
  if (v < s_diag_cdc.wa_min) s_diag_cdc.wa_min = v;
  if (v > s_diag_cdc.wa_max) s_diag_cdc.wa_max = v;
  s_diag_cdc.wa_sum += v;
  s_diag_cdc.wa_count++;
}

static void diag_cdc_on_queued(size_t q)
{
  s_diag_cdc.total_bytes_sent += q;
  s_diag_cdc.window_bytes += (uint32_t)q;
}

static void diag_cdc_maybe_log(uint64_t now_ms)
{
  const uint64_t interval_ms = (uint64_t)CONFIG_DIAG_CDC_LOG_INTERVAL_S * 1000ULL;
  if (now_ms < s_diag_cdc.window_start_ms + interval_ms) return;

  uint64_t elapsed = now_ms - s_diag_cdc.window_start_ms;
  float bps = elapsed ? (s_diag_cdc.window_bytes * 1000.0f) / (float)elapsed : 0.0f;
  uint32_t wa_avg = s_diag_cdc.wa_count ? (uint32_t)(s_diag_cdc.wa_sum / s_diag_cdc.wa_count) : 0;

  ESP_LOGI(TAG, "CDC diag: bytes_sec=%.1f B/s window=%llums total=%llu bytes, wa[min/avg/max]=%u/%u/%u count=%u",
           bps, (unsigned long long)elapsed,
           (unsigned long long)s_diag_cdc.total_bytes_sent,
           (unsigned)s_diag_cdc.wa_min == UINT32_MAX ? 0 : (unsigned)s_diag_cdc.wa_min,
           (unsigned)wa_avg,
           (unsigned)s_diag_cdc.wa_max,
           (unsigned)s_diag_cdc.wa_count);

  /* reset window */
  diag_cdc_reset_window(now_ms);
}
#endif /* CONFIG_DIAG_CDC_ENABLE */

/* Helper: bounded retry flush with small exponential backoff.
 * Returns true on success, false if all retries failed or port disconnected.
 * This avoids tight-loop logging from tinyusb when the host is transiently
 * unable to accept data.
 */
static bool safe_cdcacm_write_flush(tinyusb_cdcacm_itf_t port, int max_retries)
{
  if (!tud_cdc_n_connected((uint8_t)port)) {
    return false;
  }

  const int base_delay_ms = 100;
  uint64_t now_ms = 0;
  static uint64_t last_flush_log_ms = 0;

  for (int attempt = 0; attempt < max_retries; ++attempt) {
    /* Increase flush timeout to 1000ms to allow large buffers (32KB) to drain
     * at minimum USB speeds (335KB/s -> ~96ms), with margin for host latency.
     */
    esp_err_t err = tinyusb_cdcacm_write_flush(port, 1000);
    if (err == ESP_OK) {
      return true;
    }
    if (!tud_cdc_n_connected((uint8_t)port)) {
      return false;
    }
    vTaskDelay(pdMS_TO_TICKS(base_delay_ms << attempt));
  }

  /* Rate-limit the warning to once per second to avoid spam. */
  now_ms = esp_timer_get_time() / 1000ULL;
  if (now_ms > last_flush_log_ms + 1000ULL) {
    ESP_LOGW(TAG, "CDC flush failed after %d retries", max_retries);
    last_flush_log_ms = now_ms;
  }
  return false;
}

static void cdc_tx_task(void *arg) {
#if 1
  static frame_wire_t frm;

  const size_t max_chunk = 512;
  const uint32_t wait_step_ms = 2;
  /* Increase timeout significantly for large FFT sizes to prevent frame drops
   * that cause stream desynchronization. For FFT=4096, frame interval is ~100ms;
   * allow up to 10 seconds for host processing/backpressure.
   */
  const uint32_t wait_budget_ms = 10000;

  
#if CONFIG_DIAG_CDC_ENABLE
  /* initialize diagnostic window at task start */
  diag_cdc_reset_window(esp_timer_get_time() / 1000ULL);
  uint64_t last_stack_log_ms = esp_timer_get_time() / 1000ULL;
#endif
  while (1) {
    if (xQueueReceive(g_frameq, &frm, portMAX_DELAY) == pdTRUE) {
      if (!tusb_cdc_acm_initialized(g_data_port) ||
          !tud_cdc_n_connected((uint8_t)g_data_port)) {
        /* Host not connected: sleep longer to avoid starving idle/other tasks
         */
        vTaskDelay(pdMS_TO_TICKS(100));
        continue;
      }

      /* Send in bounded chunks, waiting briefly for space if needed. */
      size_t sent = 0;
      uint32_t waited_ms = 0;
      while (sent < FRAME_TOTAL_SIZE) {
        if (!tud_cdc_n_connected((uint8_t)g_data_port)) {
          break;
        }

        size_t chunk = FRAME_TOTAL_SIZE - sent;
        if (chunk > max_chunk) {
          chunk = max_chunk;
        }

        uint32_t wa = tud_cdc_n_write_available((uint8_t)g_data_port);
#if CONFIG_DIAG_CDC_ENABLE && CONFIG_DIAG_CDC_SAMPLE_WRITE_AVAILABLE
        diag_cdc_sample_wa(wa);
#endif
        if (wa < chunk) {
          if (waited_ms >= wait_budget_ms) {
            ESP_LOGW(TAG, "CDC TX backpressure (%u ms), dropping frame",
                     (unsigned)waited_ms);
            break;
          }
          vTaskDelay(pdMS_TO_TICKS(wait_step_ms));
          waited_ms += wait_step_ms;
          continue;
        }

        size_t queued =
            tinyusb_cdcacm_write_queue(g_data_port, frm.bytes + sent, chunk);
        if (queued == 0) {
          if (waited_ms >= wait_budget_ms) {
            ESP_LOGW(TAG, "CDC TX stalled (%u ms), dropping frame",
                     (unsigned)waited_ms);
            break;
          }
          vTaskDelay(pdMS_TO_TICKS(wait_step_ms));
          waited_ms += wait_step_ms;
          continue;
        }
#if CONFIG_DIAG_CDC_ENABLE
        diag_cdc_on_queued(queued);
#endif
        sent += queued;
      }

      if (sent < FRAME_TOTAL_SIZE) {
        ESP_LOGW(TAG, "CDC send dropped %u bytes",
                 (unsigned)(FRAME_TOTAL_SIZE - sent));
#if CONFIG_DIAG_CDC_ENABLE
        /* Report diagnostics even when a frame is dropped. */
        uint64_t now_ms = esp_timer_get_time() / 1000ULL;
        diag_cdc_maybe_log(now_ms);
#endif
        continue;
      }

      if (!safe_cdcacm_write_flush(g_data_port, 6)) {
        ESP_LOGD(TAG, "CDC flush timeout (non-fatal)");
      }
#if CONFIG_STATUS_NEOPIXEL_ENABLE
      if (g_led_strip) {
        if (!g_led_lock ||
            xSemaphoreTake(g_led_lock, pdMS_TO_TICKS(10)) == pdTRUE) {
          uint8_t b = (uint8_t)CONFIG_STATUS_NEOPIXEL_BRIGHTNESS;
          ESP_ERROR_CHECK(led_strip_set_pixel(g_led_strip, 0, 0, b, 0));
          ESP_ERROR_CHECK(led_strip_refresh(g_led_strip));
          if (g_led_lock) {
            xSemaphoreGive(g_led_lock);
          }
        }
      }
#endif
#if CONFIG_DIAG_CDC_ENABLE
      /* periodic summary log (non-blocking) */
      uint64_t now_ms = esp_timer_get_time() / 1000ULL;
      diag_cdc_maybe_log(now_ms);

      if (now_ms > last_stack_log_ms + 5000ULL) {
        UBaseType_t cdc_hwm = uxTaskGetStackHighWaterMark(g_cdc_task_handle);
        UBaseType_t adc_hwm = uxTaskGetStackHighWaterMark(g_adc_task_handle);
        ESP_LOGI(TAG, "Stack HWM: cdc_tx=%lu words, adc_fft=%lu words",
                 (unsigned long)cdc_hwm, (unsigned long)adc_hwm);
        last_stack_log_ms = now_ms;
      }
#endif
    }
  }
}
#endif

#if CONFIG_STATUS_NEOPIXEL_ENABLE
static void status_led_init(void) {
  g_led_lock = xSemaphoreCreateMutex();
  led_strip_config_t strip_config = {
      .strip_gpio_num = CONFIG_STATUS_NEOPIXEL_GPIO,
      .max_leds = 1,
      .led_pixel_format = LED_PIXEL_FORMAT_GRB,
      .led_model = LED_MODEL_WS2812,
      .flags.invert_out = false,
  };
  led_strip_rmt_config_t rmt_config = {
      .clk_src = RMT_CLK_SRC_DEFAULT,
      .resolution_hz = LED_STRIP_RMT_RES_HZ,
      .flags.with_dma = false,
  };
  ESP_ERROR_CHECK(
      led_strip_new_rmt_device(&strip_config, &rmt_config, &g_led_strip));
}

static void status_led_set(uint8_t r, uint8_t g, uint8_t b) {
  if (!g_led_strip) {
    return;
  }
  if (g_led_lock && xSemaphoreTake(g_led_lock, pdMS_TO_TICKS(10)) != pdTRUE) {
    return;
  }
  uint8_t scale = (uint8_t)CONFIG_STATUS_NEOPIXEL_BRIGHTNESS;
  uint8_t r_s = (uint8_t)((r * scale) / 255u);
  uint8_t g_s = (uint8_t)((g * scale) / 255u);
  uint8_t b_s = (uint8_t)((b * scale) / 255u);
  ESP_ERROR_CHECK(led_strip_set_pixel(g_led_strip, 0, r_s, g_s, b_s));
  ESP_ERROR_CHECK(led_strip_refresh(g_led_strip));
  if (g_led_lock) {
    xSemaphoreGive(g_led_lock);
  } 
}
#endif

bool adc_calibration_init(adc_unit_t unit, adc_atten_t atten) {
  adc_cali_curve_fitting_config_t cali_config = {
      .unit_id = unit,
      .atten = atten,
      .bitwidth = ADC_BITWIDTH_12,
  };
  if (adc_cali_create_scheme_curve_fitting(&cali_config, &cali_handle) ==
      ESP_OK) {
    ESP_LOGI(TAG, "ADC calibration ready (curve fitting)");
    return true;
  }
  ESP_LOGW(TAG, "ADC calibration not available; raw counts will be used");
  return false;
}

static inline float sample_to_volts(uint16_t raw, bool calibrated) {
  if (calibrated) {
    int mv = 0;
    if (adc_cali_raw_to_voltage(cali_handle, raw, &mv) == ESP_OK) {
      return (float)mv / 1000.0f;
    }
  }
  return (float)raw; // fallback raw counts
}

static inline float sample_to_centered(uint16_t raw, bool calibrated,
                                       int midpoint_mv) {
  if (calibrated) {
    int mv = 0;
    if (adc_cali_raw_to_voltage(cali_handle, raw, &mv) == ESP_OK) {
      return ((float)(mv - midpoint_mv)) / 1000.0f;
    }
  }
  return (float)raw - 2048.0f; // 12-bit midpoint
}

OPTIMIZE_O2 static void apply_window(float *buf) {
  for (int i = 0; i < FFT_SIZE; i++) {
    buf[i] *= win[i];
  }
}

OPTIMIZE_O2 static void perform_fft(float *buf) {
  dsps_fft4r_fc32(buf, FFT_SIZE >> 1);
  dsps_bit_rev4r_fc32(buf, FFT_SIZE >> 1);
  dsps_cplx2real_fc32(buf, FFT_SIZE >> 1);
}

OPTIMIZE_O2 static void apply_complex_cutoff(float *buf, uint32_t sps) {
#if CONFIG_OUTPUT_CUTOFF_HZ > 0
  uint32_t cutoff_hz = (uint32_t)CONFIG_OUTPUT_CUTOFF_HZ;
  uint32_t max_bin = (cutoff_hz * (uint64_t)FFT_SIZE) / sps;
  if (max_bin >= FFT_SIZE / 2) {
    return;
  }
  if (max_bin < FFT_SIZE / 2) {
    buf[1] = 0.0f; // Nyquist packed at index 1
  }
  for (uint32_t k = max_bin + 1; k < FFT_SIZE / 2; k++) {
    buf[2 * k] = 0.0f;
    buf[2 * k + 1] = 0.0f;
  }
#endif
}

static adc_atten_t cfg_atten(void) {
#if CONFIG_ADC_ATTEN_DB_0
  return ADC_ATTEN_DB_0;
#elif CONFIG_ADC_ATTEN_DB_2_5
  return ADC_ATTEN_DB_2_5;
#elif CONFIG_ADC_ATTEN_DB_6
  return ADC_ATTEN_DB_6;
#else
  return ADC_ATTEN_DB_12;
#endif
}

static void adc_fft_task(void *arg) {
  adc_atten_t atten = cfg_atten();
  bool calibrated = adc_calibration_init(ADC_UNIT_1, atten);
  int midpoint_mv = 0;
  if (calibrated) {
    (void)adc_cali_raw_to_voltage(cali_handle, 2048, &midpoint_mv);
  }

  /* ADC decode diagnostics: validates that DMA words are parsed as expected
   * (unit/channel/bitwidth). If ADC output format mismatches parsing, these
   * counters will look obviously wrong (e.g., few accepted samples, many
   * skipped/other channels).
   */
  uint64_t last_adc_diag_ms = esp_timer_get_time() / 1000ULL;
  uint32_t diag_reads = 0;
  uint32_t diag_out_bytes = 0;
  uint32_t diag_accept_ch0 = 0;
  uint32_t diag_accept_ch1 = 0;
  uint32_t diag_skip_unit = 0;
  uint32_t diag_other_chan = 0;

  adc_continuous_handle_t adc_handle = NULL;
  adc_continuous_handle_cfg_t handle_cfg = {
      .max_store_buf_size =
          FFT_SIZE * CHANNELS * sizeof(adc_digi_output_data_t),
      .conv_frame_size = 512,
  };
  ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &adc_handle));

  uint32_t sample_rate = ADC_SPS;
#if defined(CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_HIGH)
  if (sample_rate > CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_HIGH) {
    ESP_LOGW(TAG, "ADC_SPS=%u above SOC max %u; clamping", sample_rate,
             (unsigned)CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_HIGH);
    sample_rate = CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_HIGH;
  }
#endif
#if defined(CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_LOW)
  if (sample_rate < CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_LOW) {
    ESP_LOGW(TAG, "ADC_SPS=%u below SOC min %u; clamping", sample_rate,
             (unsigned)CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_LOW);
    sample_rate = CONFIG_SOC_ADC_SAMPLE_FREQ_THRES_LOW;
  }
#endif

  /* Per-channel sampling rate: adc_continuous divides sample_freq_hz by the
   * pattern length. With two channels, each gets half the configured rate. */
  const uint32_t per_channel_sps = sample_rate / CHANNELS;

  adc_digi_pattern_config_t pattern[CHANNELS] = {0};
  pattern[0].atten = atten;
  pattern[0].channel = CONFIG_ADC_CH0;
  pattern[0].unit = ADC_UNIT_1;
  pattern[0].bit_width = ADC_BITWIDTH_12;
  pattern[1].atten = atten;
  pattern[1].channel = CONFIG_ADC_CH1;
  pattern[1].unit = ADC_UNIT_1;
  pattern[1].bit_width = ADC_BITWIDTH_12;

  adc_continuous_config_t dig_cfg = {
      .sample_freq_hz = sample_rate,
      .conv_mode = ADC_CONV_SINGLE_UNIT_1,
      /* Must match adc_digi_output_data_t parsing below (p->type2.*). */
      .format = ADC_DIGI_OUTPUT_FORMAT_TYPE2,
      .pattern_num = CHANNELS,
      .adc_pattern = pattern,
  };
  ESP_ERROR_CHECK(adc_continuous_config(adc_handle, &dig_cfg));

#if CONFIG_ADC_IIR_FILTER_ENABLE
  adc_digi_iir_filter_t coeff = ADC_DIGI_IIR_FILTER_COEFF_2;
#if CONFIG_ADC_IIR_FILTER_COEFF_4
  coeff = ADC_DIGI_IIR_FILTER_COEFF_4;
#elif CONFIG_ADC_IIR_FILTER_COEFF_8
  coeff = ADC_DIGI_IIR_FILTER_COEFF_8;
#elif CONFIG_ADC_IIR_FILTER_COEFF_16
  coeff = ADC_DIGI_IIR_FILTER_COEFF_16;
#elif CONFIG_ADC_IIR_FILTER_COEFF_64
  coeff = ADC_DIGI_IIR_FILTER_COEFF_64;
#endif
  adc_continuous_iir_filter_config_t filter_cfg = {
      .unit = ADC_UNIT_1,
      .coeff = coeff,
  };
  adc_iir_filter_handle_t iir_handle = NULL;
  ESP_ERROR_CHECK(
      adc_new_continuous_iir_filter(adc_handle, &filter_cfg, &iir_handle));
  ESP_ERROR_CHECK(adc_continuous_iir_filter_enable(iir_handle));
  ESP_LOGI(TAG, "ADC IIR Filter enabled");
#endif

  ESP_ERROR_CHECK(adc_continuous_start(adc_handle));

  ESP_LOGI(TAG,
           "Sampling ADC1 CH%d (GPIO%d) and CH%d (GPIO%d) at %u sps/ch, FFT %d",
           CONFIG_ADC_CH0, CONFIG_ADC_CH0 + 1, CONFIG_ADC_CH1,
           CONFIG_ADC_CH1 + 1, (unsigned)per_channel_sps, FFT_SIZE);

  const size_t read_len = FFT_SIZE * CHANNELS * sizeof(adc_digi_output_data_t);
  uint8_t *raw = heap_caps_aligned_alloc(16, read_len,
                                         MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
  if (!raw) {
    raw = heap_caps_aligned_alloc(16, read_len, MALLOC_CAP_DMA);
  }
  if (!raw) {
    ESP_LOGE(TAG, "Failed to reserve DMA buffer (%zu bytes)", read_len);
    vTaskDelete(NULL);
    return;
  }

  static frame_wire_t frame;
  while (1) {
#if CONFIG_STATUS_NEOPIXEL_ENABLE
    status_led_set(0, 0, 255);
#endif
    int count0 = 0;
    int count1 = 0;
    double sum0 = 0;
    double sum1 = 0;
    while (count0 < FFT_SIZE || count1 < FFT_SIZE) {
      uint32_t out_len = 0;
      esp_err_t ret = adc_continuous_read(adc_handle, raw, read_len, &out_len,
                                          pdMS_TO_TICKS(100));
      if (ret == ESP_ERR_TIMEOUT) {
        continue;
      }
      if (ret != ESP_OK) {
        ESP_LOGW(TAG, "adc_continuous_read returned %s", esp_err_to_name(ret));
        vTaskDelay(pdMS_TO_TICKS(1));
        continue;
      }

      /* Per-read diagnostics. */
      diag_reads++;
      diag_out_bytes += out_len;

      for (uint32_t i = 0; i + sizeof(adc_digi_output_data_t) <= out_len;
           i += sizeof(adc_digi_output_data_t)) {
        adc_digi_output_data_t *p = (adc_digi_output_data_t *)&raw[i];
        if (p->type2.unit != ADC_UNIT_1) {
          diag_skip_unit++;
          continue;
        }
        if (p->type2.channel == CONFIG_ADC_CH0 && count0 < FFT_SIZE) {
          float val = sample_to_centered(p->type2.data, calibrated, midpoint_mv);
          ch0[count0++] = val;
          sum0 += val;
          diag_accept_ch0++;
        } else if (p->type2.channel == CONFIG_ADC_CH1 && count1 < FFT_SIZE) {
          float val = sample_to_centered(p->type2.data, calibrated, midpoint_mv);
          ch1[count1++] = val;
          sum1 += val;
          diag_accept_ch1++;
        } else {
          diag_other_chan++;
        }
      }

      /* Rate-limit to avoid log spam. */
      uint64_t now_ms = esp_timer_get_time() / 1000ULL;
      if (now_ms > last_adc_diag_ms + 2000ULL) {
        ESP_LOGI(TAG,
                 "ADC diag (2s): reads=%u out=%uB accept[ch0/ch1]=%u/%u skip_unit=%u other_chan=%u",
                 (unsigned)diag_reads, (unsigned)diag_out_bytes,
                 (unsigned)diag_accept_ch0, (unsigned)diag_accept_ch1,
                 (unsigned)diag_skip_unit, (unsigned)diag_other_chan);
        diag_reads = 0;
        diag_out_bytes = 0;
        diag_accept_ch0 = 0;
        diag_accept_ch1 = 0;
        diag_skip_unit = 0;
        diag_other_chan = 0;
        last_adc_diag_ms = now_ms;
      }
    }

    /* Dynamic DC bias removal: subtract current block mean */
    float mean0 = (float)(sum0 / FFT_SIZE);
    float mean1 = (float)(sum1 / FFT_SIZE);
    for (int i = 0; i < FFT_SIZE; i++) {
      ch0[i] -= mean0;
      ch1[i] -= mean1;
    }

#if CONFIG_ADC_ENABLE_WINDOW
    apply_window(ch0);
    apply_window(ch1);
#endif
    perform_fft(ch0);
    perform_fft(ch1);

   // apply_complex_cutoff(ch0, per_channel_sps);
   // apply_complex_cutoff(ch1, per_channel_sps);

    uint64_t now_ms = esp_timer_get_time() / 1000ULL;

    build_frame(&frame, ch0, "ch0", now_ms, per_channel_sps, g_seq++);
    if (xQueueSend(g_frameq, &frame, portMAX_DELAY) != pdTRUE) {
      ESP_LOGW(TAG, "Frame queue send failed (ch0)");
    }

    build_frame(&frame, ch1, "ch1", now_ms, per_channel_sps, g_seq++);
    if (xQueueSend(g_frameq, &frame, portMAX_DELAY) != pdTRUE) {
      ESP_LOGW(TAG, "Frame queue send failed (ch1)");
    }
  }
}

#if CONFIG_ENABLE_I2S_WAVE_GEN
static void sweep_task(void *arg) {
  const float f_start = 33.0f;
  const float f_end = 3333.0f;
  const float duration_sec = 9.0f;
  const int update_ms = 50;
  
  // Linear sweep step
  const float step = (f_end - f_start) / (duration_sec * (1000.0f / update_ms));
  
  float freq = f_start;
  while (1) {
    wave_gen_set_freq(freq);
    vTaskDelay(pdMS_TO_TICKS(update_ms));
    
    freq += step;
    if (freq > f_end) {
      freq = f_start;
    }
  }
}
#endif

void app_main(void) {
#if CONFIG_STATUS_NEOPIXEL_ENABLE
  status_led_init();
  status_led_set(255, 0, 0);
#endif
  tinyusb_config_t tusb_cfg = {
      .device_descriptor = NULL,
      .string_descriptor = NULL,
      .external_phy = false,
  };
  ESP_ERROR_CHECK(tinyusb_driver_install(&tusb_cfg));

#if CONFIG_TINYUSB_CDC_COUNT > 1
  g_data_port = TINYUSB_CDC_ACM_1;
#else
  g_data_port = TINYUSB_CDC_ACM_0;
#endif

  tinyusb_config_cdcacm_t cdc_cfg_data = {
      .usb_dev = TINYUSB_USBDEV_0,
      .cdc_port = g_data_port,
      .rx_unread_buf_sz = CONFIG_TINYUSB_CDC_RX_BUFSIZE,
      .callback_rx = NULL,
      .callback_rx_wanted_char = NULL,
      .callback_line_state_changed = NULL,
      .callback_line_coding_changed = NULL,
  };
  ESP_ERROR_CHECK(tusb_cdc_acm_init(&cdc_cfg_data));


#if CONFIG_ENABLE_I2S_WAVE_GEN
  // Initialize Wave Gen
  // Using GPIO 15-18 to avoid VDD_SPI (GPIO 11) and ADC1 interference
  wave_gen_config_t wg_cfg = {
      .mck_io_num = -1, // SCK disconnected (PCM5102 internal generation)
      .bck_io_num = 16,
      .ws_io_num = 17,
      .data_out_num = 18,
      .sample_rate = 44100,
  };
  ESP_ERROR_CHECK(wave_gen_init(&wg_cfg));
  // wave_gen_set_freq(33.0f); // Controlled by sweep_task
  wave_gen_set_type(WAVE_TYPE_SINE);
  wave_gen_set_volume(0.5f);
  ESP_ERROR_CHECK(wave_gen_start());
#endif

#if CONFIG_ENABLE_SDM_WAVE_GEN
  ESP_ERROR_CHECK(wave_gen_sdm_start());
#endif

  /* Initialize DSP before any task can call FFT/window code. */
  ESP_ERROR_CHECK(dsps_fft4r_init_fc32(NULL, FFT_SIZE >> 1));
#if CONFIG_ADC_ENABLE_WINDOW
#if CONFIG_ADC_WINDOW_HANN
  dsps_wind_hann_f32(win, FFT_SIZE);
#elif CONFIG_ADC_WINDOW_BLACKMAN
  dsps_wind_blackman_f32(win, FFT_SIZE);
#elif CONFIG_ADC_WINDOW_BLACKMAN_HARRIS
  dsps_wind_blackman_harris_f32(win, FFT_SIZE);
#elif CONFIG_ADC_WINDOW_BLACKMAN_NUTTALL
  dsps_wind_blackman_nuttall_f32(win, FFT_SIZE);
#elif CONFIG_ADC_WINDOW_NUTTALL
  dsps_wind_nuttall_f32(win, FFT_SIZE);
#elif CONFIG_ADC_WINDOW_FLAT_TOP
  dsps_wind_flat_top_f32(win, FFT_SIZE);
#else
  for (int i = 0; i < FFT_SIZE; i++) {
    win[i] = 1.0f;
  }
#endif
#endif


#if CONFIG_ENABLE_I2S_WAVE_GEN
  xTaskCreate(sweep_task, "sweep", 2048, NULL, 5, NULL);
#endif

  g_frameq = xQueueCreate(
                          2,
                          sizeof(frame_wire_t));
  assert(g_frameq);

  const BaseType_t tusb_core = 0;
#if CONFIG_FREERTOS_UNICORE
  const BaseType_t adc_core = 0;
#else
  const BaseType_t adc_core = 1;
#endif

  xTaskCreatePinnedToCore(
                          cdc_tx_task,
                          "cdc_tx",
                          CDC_TX_TASK_STACK_SIZE,
                          NULL,
                          8,
                          &g_cdc_task_handle,
                          tusb_core);

  xTaskCreatePinnedToCore(
                          adc_fft_task,
                          "adc_fft",

                          ADC_TASK_STACK_SIZE,
                          NULL,
                          8,
                          &g_adc_task_handle,
                          adc_core);

  // Tasks run indefinitely; app_main can return.
}


