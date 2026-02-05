#include <math.h>
#include <string.h>
#include "sdkconfig.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_check.h"
#include "wave_gen.h"

#if CONFIG_ENABLE_SDM_WAVE_GEN
#if defined(__has_include)
#if __has_include("driver/sdm.h") && __has_include("driver/gptimer.h")
#define WAVE_GEN_HAS_SDM 1
#include "driver/sdm.h"
#include "driver/gptimer.h"
#include "esp_attr.h"
#else
#define WAVE_GEN_HAS_SDM 0
#endif
#else
#define WAVE_GEN_HAS_SDM 1
#include "driver/sdm.h"
#include "driver/gptimer.h"
#include "esp_attr.h"
#endif
#else
#define WAVE_GEN_HAS_SDM 0
#endif

static const char *TAG = "wave_gen";

#define I2S_BUFFER_SIZE     1024
#define DEFAULT_SAMPLE_RATE 44100
#define I2S_BUFFER_SIZE     1024
#define DEFAULT_SAMPLE_RATE 44100
#define PI                  3.14159265358979323846f

static i2s_chan_handle_t tx_conn_handle = NULL;
static TaskHandle_t s_wave_task_handle = NULL;
static bool s_running = false;

// Generator State
static float s_frequency = 440.0f;
static float s_volume = 0.5f;
static wave_type_t s_type = WAVE_TYPE_SINE;
static uint32_t s_sample_rate = DEFAULT_SAMPLE_RATE;
static float s_phase = 0.0f;

#if WAVE_GEN_HAS_SDM
#define SDM_LUT_SIZE 256
#define SDM_CARRIER_HZ (1 * 1000 * 1000)
#define SDM_UPDATE_HZ (20 * 1000)
#define SDM_DENSITY_MAX 90

static sdm_channel_handle_t s_sdm_chan = NULL;
static gptimer_handle_t s_sdm_timer = NULL;
static volatile uint32_t s_sdm_phase_acc = 0;
static uint32_t s_sdm_phase_step = 0;
static int8_t s_sdm_lut[SDM_LUT_SIZE];
static bool s_sdm_running = false;

static void sdm_build_lut(float amplitude) {
    if (amplitude < 0.0f) amplitude = 0.0f;
    if (amplitude > 1.0f) amplitude = 1.0f;
    for (int i = 0; i < SDM_LUT_SIZE; i++) {
        float phase = (float)i / (float)SDM_LUT_SIZE;
        float s = sinf(phase * 2.0f * PI);
        int32_t density = (int32_t)lrintf(s * amplitude * (float)SDM_DENSITY_MAX);
        if (density > SDM_DENSITY_MAX) density = SDM_DENSITY_MAX;
        if (density < -SDM_DENSITY_MAX) density = -SDM_DENSITY_MAX;
        s_sdm_lut[i] = (int8_t)density;
    }
}

static bool IRAM_ATTR sdm_timer_cb(gptimer_handle_t timer,
                                  const gptimer_alarm_event_data_t *edata,
                                  void *user_ctx) {
    (void)timer;
    (void)edata;
    (void)user_ctx;
    uint32_t phase = s_sdm_phase_acc + s_sdm_phase_step;
    s_sdm_phase_acc = phase;
    uint8_t idx = (uint8_t)(phase >> 24);
    sdm_channel_set_pulse_density(s_sdm_chan, s_sdm_lut[idx]);
    return false;
}

static esp_err_t sdm_init(uint32_t freq_hz) {
    if (s_sdm_chan) {
        return ESP_OK;
    }

    sdm_config_t config = {
        .clk_src = SDM_CLK_SRC_DEFAULT,
        .sample_rate_hz = SDM_CARRIER_HZ,
        .gpio_num = CONFIG_SDM_GPIO,
    };
    ESP_RETURN_ON_ERROR(sdm_new_channel(&config, &s_sdm_chan), TAG, "Failed to create SDM channel");
    ESP_RETURN_ON_ERROR(sdm_channel_enable(s_sdm_chan), TAG, "Failed to enable SDM channel");

    sdm_build_lut(s_volume);
    if (freq_hz < 1) freq_hz = 1;
    s_sdm_phase_step = (uint32_t)(((uint64_t)freq_hz << 32) / SDM_UPDATE_HZ);
    s_sdm_phase_acc = 0;

    gptimer_config_t tcfg = {
        .clk_src = GPTIMER_CLK_SRC_DEFAULT,
        .direction = GPTIMER_COUNT_UP,
        .resolution_hz = SDM_UPDATE_HZ,
    };
    ESP_RETURN_ON_ERROR(gptimer_new_timer(&tcfg, &s_sdm_timer), TAG, "Failed to create SDM timer");

    gptimer_event_callbacks_t cbs = {
        .on_alarm = sdm_timer_cb,
    };
    ESP_RETURN_ON_ERROR(gptimer_register_event_callbacks(s_sdm_timer, &cbs, NULL), TAG, "Failed to register SDM timer callback");

    gptimer_alarm_config_t alarm_cfg = {
        .reload_count = 0,
        .alarm_count = 1,
        .flags.auto_reload_on_alarm = true,
    };
    ESP_RETURN_ON_ERROR(gptimer_set_alarm_action(s_sdm_timer, &alarm_cfg), TAG, "Failed to set SDM timer alarm");
    ESP_RETURN_ON_ERROR(gptimer_enable(s_sdm_timer), TAG, "Failed to enable SDM timer");
    ESP_RETURN_ON_ERROR(gptimer_start(s_sdm_timer), TAG, "Failed to start SDM timer");

    return ESP_OK;
}

static void sdm_deinit(void) {
    if (s_sdm_timer) {
        gptimer_stop(s_sdm_timer);
        gptimer_disable(s_sdm_timer);
        gptimer_del_timer(s_sdm_timer);
        s_sdm_timer = NULL;
    }
    if (s_sdm_chan) {
        sdm_channel_disable(s_sdm_chan);
        sdm_del_channel(s_sdm_chan);
        s_sdm_chan = NULL;
    }
}
#endif

static void wave_gen_task(void *args)
{
    size_t w_bytes = 0;
    // Stereo buffer (Left + Right interleaving)
    // Using 32-bit samples for high resolution
    int32_t *samples = (int32_t *)malloc(I2S_BUFFER_SIZE * 2 * sizeof(int32_t));
    if (!samples) {
        ESP_LOGE(TAG, "Failed to allocate audio buffer");
        vTaskDelete(NULL);
        return;
    }

    float phase_inc = 0.0f;
    
    while (s_running) {
        // Recalculate phase increment per block to allow dynamic frequency changes
        // phase goes from 0.0f to 1.0f
        if (s_sample_rate > 0) {
            phase_inc = s_frequency / (float)s_sample_rate;
        }

        // Fill buffer
        for (int i = 0; i < I2S_BUFFER_SIZE; i++) {
            float sample_val = 0.0f;

            switch (s_type) {
                case WAVE_TYPE_SINE:
                    sample_val = sinf(s_phase * 2.0f * PI);
                    break;
                case WAVE_TYPE_SQUARE:
                    sample_val = (s_phase < 0.5f) ? 1.0f : -1.0f;
                    break;
                case WAVE_TYPE_TRIANGLE:
                    sample_val = (s_phase < 0.5f) ? (4.0f * s_phase - 1.0f) : (3.0f - 4.0f * s_phase);
                    break;
                case WAVE_TYPE_SAWTOOTH:
                    sample_val = 2.0f * s_phase - 1.0f;
                    break;
            }

            // Apply volume and scale to 32-bit signed integer range
            // Max positive: 2147483647
            int32_t pcm_val = (int32_t)(sample_val * s_volume * 2147483647.0f);

            // Interleaved Stereo (Left = Right)
            samples[i * 2] = pcm_val;
            samples[i * 2 + 1] = pcm_val;

            // Advance phase
            s_phase += phase_inc;
            if (s_phase >= 1.0f) {
                s_phase -= 1.0f;
            }
        }

        // Write to I2S
        /* Because we are writing to a standard I2S channel, we should ensure the format matches the config */
        if (i2s_channel_write(tx_conn_handle, samples, I2S_BUFFER_SIZE * 2 * sizeof(int32_t), &w_bytes, portMAX_DELAY) != ESP_OK) {
            ESP_LOGW(TAG, "I2S Write Failed");
        }
    }

    free(samples);
    vTaskDelete(NULL);
    s_wave_task_handle = NULL;
}

esp_err_t wave_gen_init(const wave_gen_config_t *config)
{
    if (tx_conn_handle) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Initializing I2S Wave Gen (32-bit) on BCK:%d WS:%d DO:%d MCK:%d",
             config->bck_io_num, config->ws_io_num, config->data_out_num, config->mck_io_num);

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    ESP_RETURN_ON_ERROR(i2s_new_channel(&chan_cfg, &tx_conn_handle, NULL), TAG, "Failed to create I2S channel");

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(config->sample_rate),
        .slot_cfg = I2S_STD_PCM_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = config->mck_io_num, // Can be I2S_GPIO_UNUSED (-1)
            .bclk = config->bck_io_num,
            .ws = config->ws_io_num,
            .dout = config->data_out_num,
            .din = I2S_GPIO_UNUSED,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    
    // Some PCM5102 modules benefit from MCLK being enabled if wired, 
    // but often they generate it internally from BCK if MCLK is floating/grounded.
    // The driver will handle pin configuration.

    ESP_RETURN_ON_ERROR(i2s_channel_init_std_mode(tx_conn_handle, &std_cfg), TAG, "Failed to init I2S std mode");

    s_sample_rate = config->sample_rate;
    return ESP_OK;
}

esp_err_t wave_gen_start(void)
{
    if (!tx_conn_handle) return ESP_FAIL;
    if (s_running) return ESP_OK;

    ESP_RETURN_ON_ERROR(i2s_channel_enable(tx_conn_handle), TAG, "Failed to enable channel");
    
    s_running = true;
    xTaskCreate(wave_gen_task, "wave_gen_task", 4096, NULL, 5, &s_wave_task_handle);
    ESP_LOGI(TAG, "Wave Gen Started");
    return ESP_OK;
}

esp_err_t wave_gen_stop(void)
{
    if (!s_running) return ESP_OK;
    s_running = false;
    // Wait for task or just let it spin down? 
    // For simplicity, we assume task sees s_running false and exits.
    // Ideally we'd join or wait.
    
    // Allow task to exit loop
    vTaskDelay(pdMS_TO_TICKS(100)); 

    if (tx_conn_handle) {
        i2s_channel_disable(tx_conn_handle);
    }
    ESP_LOGI(TAG, "Wave Gen Stopped");
    return ESP_OK;
}

void wave_gen_set_freq(float freq_hz)
{
    if (freq_hz < 1.0f) freq_hz = 1.0f;
    if (freq_hz > (float)s_sample_rate / 2.0f) freq_hz = (float)s_sample_rate / 2.0f;
    s_frequency = freq_hz;
}

void wave_gen_set_type(wave_type_t type)
{
    s_type = type;
}

void wave_gen_set_volume(float volume)
{
    if (volume < 0.0f) volume = 0.0f;
    if (volume > 1.0f) volume = 1.0f;
    s_volume = volume;
}

esp_err_t wave_gen_sdm_start(void)
{
#if WAVE_GEN_HAS_SDM
    if (s_sdm_running) {
        return ESP_OK;
    }
    ESP_LOGI(TAG, "Starting SDM sine on GPIO %d at %d Hz", CONFIG_SDM_GPIO, CONFIG_SDM_FREQ_HZ);
    ESP_RETURN_ON_ERROR(sdm_init((uint32_t)CONFIG_SDM_FREQ_HZ), TAG, "SDM init failed");
    s_sdm_running = true;
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t wave_gen_sdm_stop(void)
{
#if WAVE_GEN_HAS_SDM
    if (!s_sdm_running) {
        return ESP_OK;
    }
    s_sdm_running = false;
    sdm_deinit();
    return ESP_OK;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}
