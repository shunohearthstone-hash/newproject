#!/usr/bin/env python3
import argparse
import binascii
import csv
import struct
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, TextIO, Any

try:
    import serial  # type: ignore
except ImportError:
    print("pyserial is required: pip install pyserial", file=sys.stderr)
    sys.exit(1)

MAGIC = b"\xF0\x0D\xF0\x0D"
HEADER_LEN = 9
CRC_LEN = 4
PAYLOAD_FIXED_LEN = 8 + 4 + 4 + 4  # t_ms, sps, fft, label


def parse_frames(buffer: bytearray):
    frames = []
    while True:
        idx = buffer.find(MAGIC)
        if idx == -1:
            if len(buffer) > 3:
                del buffer[:-3]
            break
        if idx > 0:
            del buffer[:idx]
        if len(buffer) < HEADER_LEN:
            break
        version = buffer[4]
        seq = int.from_bytes(buffer[5:7], "little")
        payload_len = int.from_bytes(buffer[7:9], "little")
        frame_len = HEADER_LEN + payload_len + CRC_LEN
        if len(buffer) < frame_len:
            break
        frame = bytes(buffer[:frame_len])
        del buffer[:frame_len]
        crc_expected = int.from_bytes(frame[-CRC_LEN:], "little")
        crc_calc = binascii.crc32(frame[:-CRC_LEN]) & 0xFFFFFFFF
        if crc_calc != crc_expected:
            continue
        frames.append((version, seq, frame[HEADER_LEN:HEADER_LEN + payload_len]))
    return frames


def payload_to_row(version: int, payload: bytes):
    if len(payload) < PAYLOAD_FIXED_LEN:
        return None
    t_ms, sps, fft, label = struct.unpack_from("<QII4s", payload, 0)
    label_text = label.decode("ascii", errors="ignore").strip("\x00")
    data_bytes = payload[PAYLOAD_FIXED_LEN:]
    if len(data_bytes) % 4 != 0:
        return None
    n_floats = len(data_bytes) // 4

    if version == 1:
        if fft and n_floats != fft // 2:
            return None
    elif version == 2:
        if fft and n_floats != fft:
            return None

    # Return raw payload bytes for deterministic float32 parsing on host
    return [t_ms, sps, fft, label_text, data_bytes]


def main() -> int:
    parser = argparse.ArgumentParser(description="Read binary framed FFT data from ESP32-S3 over USB-CDC")
    parser.add_argument("port", nargs="?", default="COM8", help="Serial port (e.g. /dev/ttyACM0 or COM9)")
    parser.add_argument("--baud", type=int, default=921600, help="Baud rate (ignored by USB-CDC, kept for UART use)")
    parser.add_argument("--output", type=Path, default=Path("stream.csv"), help="Output CSV file")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of truncating")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = run until Ctrl+C)")
    parser.add_argument("--flush-every", type=int, default=50, help="Flush to disk every N frames (default: 50)")
    parser.add_argument("--plot", action="store_true", default=True, help="Live plot ch0/ch1 like liveplot.py")
    parser.add_argument("--plot-interval", type=int, default=2, help="Plot refresh interval in ms")
    parser.add_argument("--max-freq", type=float, default=22000, help="Maximum frequency to display in Hz (0 disables)")
    parser.add_argument("--separate-plots", action="store_true", default=False,
                        help="Use separate windows for time and frequency plots")
    parser.add_argument("--audio", action="store_true", default=False, help="Play audio representation of FFT magnitudes")
    parser.add_argument("--audio-rate", type=int, default=44100, help="Audio sample rate in Hz")
    parser.add_argument("--audio-block", type=int, default=2048, help="Audio block size in samples")
    parser.add_argument("--audio-gain", type=float, default=0.2, help="Audio gain scaling (0.0-1.0)")
    parser.add_argument("--audio-max", type=float, default=0.3,
                        help="Limiter ceiling for audio output (0 disables, typical 0.2-0.5)")
    parser.add_argument("--audio-normalize", action="store_true", default=True,
                        help="Normalize each audio block to --audio-max when enabled")
    parser.add_argument("--audio-channel", choices=["ch0", "ch1"], default="ch0", help="Channel to sonify")
    parser.add_argument("--no-csv", action="store_true", default=True, help="Disable CSV output (plot only)")
    parser.add_argument("--reconnect", action="store_true", default=True, help="Auto-reconnect on device reset")
    parser.add_argument("--reconnect-delay", type=float, default=0.5, help="Seconds between reconnect attempts")
    parser.add_argument("--no-plot-freq", action="store_true", help="Disable Frequency Domain plot within plotting mode")
    parser.add_argument("--no-plot-time", action="store_true", help="Disable Time Domain plot within plotting mode")
    parser.add_argument("-w", "--window", choices=['0', '1', '2', '3'], default='0', help="Windowing: 0=None, 1=Hann, 2=Hamming, 3=Blackman")
    parser.add_argument("--inspect-frames", action="store_true", help="Print frame metadata and payload sample for debugging")
    parser.add_argument("--hex-dump", action="store_true", help="When inspecting frames, print a hex dump of the payload bytes (truncated)")
    parser.add_argument("--read-chunk", type=int, default=16384,
                        help="Serial read chunk size in bytes (default: 16384)")
    parser.add_argument("--read-max", type=int, default=262144,
                        help="Max bytes to read per loop iteration (default: 262144)")
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="Seconds between host-side throughput stats (0 disables)")
    args = parser.parse_args()

    mode = "a" if args.append else "w"
    buffer = bytearray()
    last_seq = None
    dropped = 0
    
    # Thread communication
    reader_stop_event = threading.Event()
    frame_queue: queue.Queue = queue.Queue()
    
    count = 0
    start = time.time()
    last_stats_time = start
    last_stats_count = 0
    last_stats_dropped = 0

    writer: Optional[Any] = None
    file_handle: Optional[TextIO] = None
    stop_requested = False

    audio_stream = None
    audio_synth = None

    if args.audio:
        try:
            import numpy as np  # type: ignore
            import sounddevice as sd  # type: ignore
        except ImportError:
            print("audio requires numpy and sounddevice: pip install numpy sounddevice", file=sys.stderr)
            return 1

        class AudioSynth:
            def __init__(self, rate: int, gain: float, max_freq: float, max_level: float, normalize: bool):
                self.rate = rate
                self.gain = gain
                self.max_freq = max_freq
                self.max_level = max(0.0, float(max_level))
                self.normalize = bool(normalize)
                self.lock = threading.Lock()
                self.bins: Optional[list[float]] = None
                self.sps: Optional[int] = None
                self.fft: Optional[int] = None

            def update(self, bins: list[float], sps: int, fft: int) -> None:
                with self.lock:
                    self.bins = list(bins)
                    self.sps = int(sps)
                    self.fft = int(fft)

            def callback(self, outdata, frames, _time_info, _status) -> None:
                with self.lock:
                    bins = self.bins
                    sps = self.sps
                    fft = self.fft
                if not bins or not sps or not fft:
                    outdata.fill(0)
                    return

                mags = np.array(bins, dtype=np.float32)
                if mags.size == 0:
                    outdata.fill(0)
                    return

                if np.median(mags) < 0.0:
                    mags_db = np.clip(mags, -200.0, 60.0)
                    mags = np.power(10.0, mags_db / 20.0)

                n_fft = frames * 2
                out_bins = n_fft // 2 + 1
                freqs_in = np.linspace(0.0, sps / 2.0, mags.size)
                freqs_out = np.linspace(0.0, self.rate / 2.0, out_bins)
                mags_out = np.interp(freqs_out, freqs_in, mags, left=0.0, right=0.0)

                if self.max_freq and self.max_freq > 0.0:
                    mags_out[freqs_out > self.max_freq] = 0.0

                mags_out = np.nan_to_num(mags_out, nan=0.0, posinf=0.0, neginf=0.0)
                peak = float(np.max(mags_out))
                if peak > 0.0:
                    mags_out = mags_out / peak
                mags_out = mags_out * float(self.gain)

                phases = np.random.uniform(0.0, 2.0 * np.pi, out_bins)
                phases[0] = 0.0
                if n_fft % 2 == 0:
                    phases[-1] = 0.0
                spectrum = mags_out * np.exp(1j * phases)
                audio = np.fft.irfft(spectrum, n=n_fft)[:frames]
                audio = audio.astype(np.float32)

                if self.max_level > 0.0:
                    peak = float(np.max(np.abs(audio)))
                    if self.normalize and peak > 0.0:
                        scale = self.max_level / peak
                        if scale < 1.0:
                            audio = audio * scale
                    else:
                        audio = np.clip(audio, -self.max_level, self.max_level)

                outdata[:, 0] = audio

        audio_synth = AudioSynth(
            args.audio_rate,
            args.audio_gain,
            args.max_freq,
            args.audio_max,
            args.audio_normalize,
        )
        audio_stream = sd.OutputStream(
            samplerate=args.audio_rate,
            blocksize=args.audio_block,
            channels=1,
            callback=audio_synth.callback,
        )
        audio_stream.start()

    def handle_row(seq: int, row: list):
        nonlocal last_seq, dropped, count, stop_requested
        nonlocal last_stats_time, last_stats_count, last_stats_dropped
        if last_seq is not None:
            expected = (last_seq + 1) & 0xFFFF
            if seq != expected:
                if seq < last_seq and (last_seq - seq) > 1000:
                    last_seq = seq
                    return
                dropped += (seq - expected) & 0xFFFF
        last_seq = seq
        if writer is not None:
            writer.writerow(row)
            count += 1
            if args.flush_every and count % args.flush_every == 0:
                if file_handle is not None:
                    file_handle.flush()
        else:
            count += 1
        if args.max_frames and count >= args.max_frames:
            stop_requested = True

        if args.stats_interval and args.stats_interval > 0:
            now = time.time()
            elapsed = now - last_stats_time
            if elapsed >= args.stats_interval:
                frames_delta = count - last_stats_count
                dropped_delta = dropped - last_stats_dropped
                fps = frames_delta / elapsed if elapsed > 0 else 0.0
                buffer_len = len(buffer)
                print(
                    f"Host stats: frames={count} (+{frames_delta}), "
                    f"fps={fps:.2f}, dropped={dropped} (+{dropped_delta}), "
                    f"buffer={buffer_len} bytes"
                )
                last_stats_time = now
                last_stats_count = count
                last_stats_dropped = dropped

    try:
        if not args.no_csv:
            file_handle = args.output.open(mode, newline="")
            assert file_handle is not None
            writer = csv.writer(file_handle)

        def open_serial_port() -> Optional[serial.Serial]:
            try:
                ser = serial.Serial(args.port, args.baud, timeout=0)
                ser.write_timeout = 0
                try:
                    ser.set_buffer_size(rx_size=262144, tx_size=262144)
                except (AttributeError, ValueError, serial.SerialException):
                    pass
                print(f"Connected to {args.port}")
                return ser
            except serial.SerialException as exc:
                if args.reconnect:
                    print(f"Waiting for {args.port}: {exc}", file=sys.stderr)
                    return None
                raise

        def read_serial_nonblocking(ser_obj: serial.Serial, buf: bytearray) -> None:
            read_budget = int(args.read_max)
            while read_budget > 0:
                waiting = ser_obj.in_waiting
                if waiting <= 0:
                    break
                read_size = waiting if waiting > args.read_chunk else args.read_chunk
                if read_size > read_budget:
                    read_size = read_budget
                chunk = ser_obj.read(read_size)
                if not chunk:
                    break
                buf.extend(chunk)
                read_budget -= len(chunk)

        # Threaded Serial Reader to prevent GUI blocking from causing buffer overflows
        
        def serial_reader_thread():
            local_buffer = bytearray()
            ser = open_serial_port()
            next_reconnect = time.time()
            
            while not reader_stop_event.is_set():
                if ser is None:
                    if args.reconnect and time.time() >= next_reconnect:
                        ser = open_serial_port()
                        next_reconnect = time.time() + args.reconnect_delay
                        if ser is not None:
                            local_buffer.clear()
                            # Reset sequence tracking in main thread? 
                            # We can't access last_seq easily but main thread handles gaps.
                            # Just clearing buffer is enough.
                    else:
                        time.sleep(0.1)
                    continue

                try:
                    read_serial_nonblocking(ser, local_buffer)
                    # Parse frames off the buffer
                    new_frames = parse_frames(local_buffer)
                    for nf in new_frames:
                        frame_queue.put(nf)
                except serial.SerialException as exc:
                    print(f"Serial error: {exc}", file=sys.stderr)
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    next_reconnect = time.time() + args.reconnect_delay
                except Exception as e:
                    print(f"Reader error: {e}", file=sys.stderr)
                    time.sleep(1)

                # Avoid 100% CPU
                time.sleep(0.001)
            
            if ser:
                ser.close()

        # Start the reader thread
        t_reader = threading.Thread(target=serial_reader_thread, daemon=True)
        t_reader.start()

        if args.plot:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import matplotlib.ticker as ticker
            import numpy as np
        
            def get_window(name: str, N: int):
                if name == '1': # Hann
                    return np.hanning(N)
                elif name == '2': # Hamming
                    return np.hamming(N)
                elif name == '3': # Blackman
                    return np.blackman(N)
                return np.ones(N)

            last_bins: dict[str, Optional[list[float]]] = {"ch0": None, "ch1": None}
            plot_meta: dict[str, Any] = {"sps": None, "fft": None}

            # Setup Figures
            figs = []
            ax_time = None
            ax_freq = None
            lines = []
            line_time: dict[str, Any] = {"ch0": None, "ch1": None}
            line_freq: dict[str, Any] = {"ch0": None, "ch1": None}

            if not args.no_plot_time or not args.no_plot_freq:
                if args.separate_plots:
                    if not args.no_plot_time:
                        fig_time = plt.figure("Time Domain", figsize=(10, 5))
                        figs.append(fig_time)
                        ax_time = fig_time.add_subplot(111)
                    if not args.no_plot_freq:
                        fig_freq = plt.figure("Frequency Domain", figsize=(10, 5))
                        figs.append(fig_freq)
                        ax_freq = fig_freq.add_subplot(111)
                else:
                    fig_title = "FFT Plots"
                    fig_height = 8 if (not args.no_plot_time and not args.no_plot_freq) else 5
                    fig_main = plt.figure(fig_title, figsize=(10, fig_height))
                    figs.append(fig_main)
                    if not args.no_plot_time and not args.no_plot_freq:
                        ax_time = fig_main.add_subplot(2, 1, 1)
                        ax_freq = fig_main.add_subplot(2, 1, 2)
                    elif not args.no_plot_time:
                        ax_time = fig_main.add_subplot(1, 1, 1)
                    else:
                        ax_freq = fig_main.add_subplot(1, 1, 1)

            if ax_time is not None:
                line_time["ch0"], = ax_time.plot([], [], label="ch0")
                line_time["ch1"], = ax_time.plot([], [], label="ch1")
                lines.extend([line_time["ch0"], line_time["ch1"]])
                ax_time.set_xlabel("Time (s)")
                ax_time.set_ylabel("Amplitude")
                ax_time.set_title("Time Domain")
                ax_time.legend(loc="upper right")
                ax_time.grid(True)
                ax_time.format_coord = lambda x, y: f"Time: {x:.6f} s, Amp: {y:.4f}"

            if ax_freq is not None:
                line_freq["ch0"], = ax_freq.plot([], [], label="ch0")
                line_freq["ch1"], = ax_freq.plot([], [], label="ch1")
                lines.extend([line_freq["ch0"], line_freq["ch1"]])
                ax_freq.set_xlabel("Frequency (Hz)")
                ax_freq.set_ylabel("Magnitude (dB)")
                ax_freq.set_title("Frequency Domain")
                ax_freq.legend(loc="upper right")

                # More frequent frequency scale steps
                ax_freq.xaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
                ax_freq.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax_freq.grid(True, which='major', alpha=0.8)
                ax_freq.grid(True, which='minor', alpha=0.4, linestyle='--')
                ax_freq.format_coord = lambda x, y: f"Freq: {x:.2f} Hz, Mag: {y:.2f} dB"
            
            if not figs:
                print("Warning: Plotting enabled but both Time and Freq plots disabled.")
                return 0

            plot_meta["zoom"] = 1.0
            plot_meta["amp_scale"] = 1.0

            def on_key(event):
                # Time Domain scaling (Horizontal)
                if event.key == '+' or event.key == '=':
                    plot_meta["zoom"] = max(0.01, plot_meta["zoom"] * 0.8)
                elif event.key == '-' or event.key == '_':
                    plot_meta["zoom"] = min(1.0, plot_meta["zoom"] * 1.25)
                # Amplitude scaling (Vertical)
                elif event.key == 'up':
                    plot_meta["amp_scale"] = max(0.01, plot_meta["amp_scale"] * 0.8)
                elif event.key == 'down':
                    plot_meta["amp_scale"] = min(100.0, plot_meta["amp_scale"] * 1.25)

            for f in figs:
                f.canvas.mpl_connect('key_press_event', on_key)

            ser: Optional[serial.Serial] = open_serial_port()
            next_reconnect = time.time()

            def update(_):
                # Drain queue
                frames_local = []
                try:
                    while True:
                        frames_local.append(frame_queue.get_nowait())
                except queue.Empty:
                    pass

                for version, seq, payload in frames_local:
                    row = payload_to_row(version, payload)
                    if row is None:
                        continue
                    # Optional inspection for debugging frame contents
                    if args.inspect_frames:
                        data_bytes = payload[PAYLOAD_FIXED_LEN:]
                        n_floats = len(data_bytes) // 4
                        try:
                            import numpy as _np
                            vals = _np.frombuffer(data_bytes, dtype='<f4')
                            print(f"\n=== Frame ver={version} seq={seq} sps={row[1]} fft={row[2]} label={row[3]} n={n_floats} ===")
                            print(f"Packed format: [DC, Nyq, Re1, Im1, Re2, Im2, ...]")
                            print(f"  vals[0] (DC)     = {vals[0]:.6f}")
                            print(f"  vals[1] (Nyquist)= {vals[1]:.6f}")
                            print(f"  vals[2:6] (bins 1-2) = {vals[2:6].tolist()}")
                            if n_floats == int(row[2]):
                                # Extract Re/Im for all bins
                                re = vals[2::2]
                                im = vals[3::2]
                                # Compute magnitude for each bin
                                mags = _np.sqrt(re**2 + im**2)
                                # Find peak bin (excluding DC)
                                peak_bin = _np.argmax(mags) + 1  # +1 because re/im start at bin 1
                                peak_freq = peak_bin * int(row[1]) / int(row[2])
                                print(f"  Spectrum stats:")
                                print(f"    Re: min={float(_np.min(re)):.4f} max={float(_np.max(re)):.4f} median={float(_np.median(re)):.4f}")
                                print(f"    Im: min={float(_np.min(im)):.4f} max={float(_np.max(im)):.4f} median={float(_np.median(im)):.4f}")
                                print(f"    Mag: min={float(_np.min(mags)):.4f} max={float(_np.max(mags)):.4f}")
                                print(f"    Peak bin={peak_bin} (~{peak_freq:.1f} Hz), mag={float(mags[peak_bin-1]):.4f}")
                                # Check if data looks like complex FFT vs magnitude-only
                                im_energy = _np.sum(im**2)
                                re_energy = _np.sum(re**2)
                                if im_energy < 1e-10 and re_energy > 1e-6:
                                    print(f"  WARNING: Im energy ~0, data may be magnitude-only (not complex)!")
                            if args.hex_dump:
                                max_bytes = 128
                                show = data_bytes[:max_bytes]
                                hex_show = binascii.hexlify(show).decode('ascii')
                                grouped = " ".join(hex_show[i:i+2] for i in range(0, len(hex_show), 2))
                                if len(data_bytes) > max_bytes:
                                    grouped += " ..."
                                print(f"  Payload Hex ({len(data_bytes)} bytes): {grouped}")
                        except Exception as e:
                            # Fallback simple print if numpy unavailable
                            sample = struct.unpack_from("<%df" % min(16, n_floats), data_bytes, 0)
                            print(f"Frame ver={version} seq={seq} sps={row[1]} fft={row[2]} label={row[3]} n={n_floats}")
                            print(f"first values: {sample}")
                            print(f"Exception: {e}")
                    handle_row(seq, row)
                    if stop_requested:
                        plt.close('all')
                        return lines
                    
                    label = str(row[3])
                    # row[4] is raw bytes (float32 little-endian)
                    bins_field = row[4]
                    # Do not convert to Python floats here; store raw bytes
                    # so reconstruction uses numpy.frombuffer (no copying).
                    plot_meta["sps"] = int(row[1])
                    plot_meta["fft"] = int(row[2])

                    if "ch0" in label:
                        last_bins["ch0"] = bins_field
                    elif "ch1" in label:
                        last_bins["ch1"] = bins_field
                        
                    if args.audio and audio_synth is not None:
                        if args.audio_channel in label:
                            # Convert raw bytes to float list for audio path
                            if isinstance(bins_field, (bytes, bytearray, memoryview)):
                                try:
                                    audio_bins = np.frombuffer(bins_field, dtype='<f4').tolist()
                                except Exception:
                                    audio_bins = list(struct.unpack("<%df" % (len(bins_field)//4), bins_field))
                            else:
                                audio_bins = list(bins_field)
                            audio_synth.update(audio_bins, plot_meta["sps"], plot_meta["fft"])

                # Update Plots
                updated_time = False
                updated_freq = False
                
                last_sps = plot_meta["sps"]
                last_fft = plot_meta["fft"]
                
                if last_sps and last_fft and (last_bins["ch0"] is not None or last_bins["ch1"] is not None):
                    # Helper to process one channel
                    def process_channel(bins_list):
                        if bins_list is None: return None, None
                        # Accept either raw bytes (preferred) or Python lists
                        if isinstance(bins_list, (bytes, bytearray, memoryview)):
                            raw_in = np.frombuffer(bins_list, dtype='<f4')
                        else:
                            raw_in = np.array(bins_list, dtype=np.float32)
                        time_data = None
                        freq_data = None
                        
                        # Handle Version 1 (Real Mag only) - can't do Time, only Freq
                        if len(raw_in) == last_fft // 2: 
                            # Just Magnitude in dB? Or Linear?
                            # Original code assumed dB if median < 0. Here we assume existing logic was correct.
                            # But wait, Version 1 firmware sends log in Kconfig default.
                            freq_data = raw_in # Already mag/db
                            if ax_time: time_data = np.zeros(last_fft) # Cannot reconstruct
                        
                        # Handle Version 2 (Complex)
                        elif len(raw_in) == last_fft:
                            # 1. Unpack esp-dsp Real FFT format to Numpy Complex
                            # esp-dsp: [R0, R_N/2, R1, I1, R2, I2 ... ]
                            # numpy rfft: [R0, R1+I1j, ..., R_N/2] (Sort of, actually expects N/2+1 complex)
                            
                            # Construct complex array for IRFFT
                            # Size N//2 + 1
                            c_spec = np.zeros(last_fft // 2 + 1, dtype=np.complex128)
                            c_spec[0] = raw_in[0] # DC
                            c_spec[-1] = raw_in[1] # Nyquist
                            # Indices 2,3 -> bin 1. 
                            # raw_in has N floats. 
                            # 2..N-1 are Re/Im pairs.
                            re = raw_in[2::2]
                            im = raw_in[3::2]
                            # Assign to c_spec[1:-1]
                            L = min(len(re), len(im), len(c_spec)-2)
                            if L > 0:
                                c_spec[1:1+L] = re[:L] + 1j * im[:L]
                            
                            # 2. IFFT to get Raw Time
                            time_data = np.fft.irfft(c_spec, n=last_fft)
                            # Optional debug printing for reconstructed time data
                            if args.inspect_frames:
                                try:
                                    import numpy as _np
                                    tvals = time_data
                                    t_min, t_max = float(_np.min(tvals)), float(_np.max(tvals))
                                    t_pp = t_max - t_min  # peak-to-peak
                                    t_rms = float(_np.sqrt(_np.mean(tvals**2)))
                                    print(f"  Time-domain reconstruction:")
                                    print(f"    len={len(tvals)} min={t_min:.4f} max={t_max:.4f} p-p={t_pp:.4f} RMS={t_rms:.4f}")
                                    print(f"    first 8 samples: {tvals[:8].tolist()}")
                                    # Check for spike vs sine wave
                                    if t_pp > 0:
                                        # Count zero crossings to detect periodicity
                                        zc = _np.sum(_np.diff(_np.sign(tvals - _np.mean(tvals))) != 0)
                                        print(f"    zero crossings={zc} (expect ~{last_fft//50} for typical audio)")
                                except Exception as e:
                                    print(f"  time_data debug error: {e}")
                            
                            # 3. Apply Window (Host Side)
                            if args.window != '0':
                                win = get_window(args.window, last_fft)
                                windowed_time = time_data * win
                                # 4. FFT for Frequency Plot
                                c_windowed = np.fft.rfft(windowed_time)
                                # Compute Magnitude (Linear then dB)
                                mag_lin = np.abs(c_windowed) * (2.0 / last_fft) # Scale
                                mag_lin[0] /= 2.0 # DC handling
                                # Safe log
                                freq_data = 20 * np.log10(mag_lin + 1e-9)
                                # Cut off Nyquist from display if desired, or keep N/2
                                freq_data = freq_data[:-1] # Match N/2 bins usually 
                            else:
                                # No window: Use the magnitude from the raw input spectrum directly
                                # Re-calculate magnitude from c_spec
                                mag_lin = np.abs(c_spec) * (2.0 / last_fft)
                                mag_lin[0] /= 2.0
                                mag_lin[-1] /= 2.0 # Nyquist
                                freq_data = 20 * np.log10(mag_lin + 1e-9)
                                freq_data = freq_data[:-1] 

                        return time_data, freq_data

                    t0, f0 = process_channel(last_bins["ch0"])
                    t1, f1 = process_channel(last_bins["ch1"])
                    
                    # Update Time Plot
                    if ax_time and line_time["ch0"] is not None and line_time["ch1"] is not None:
                        t_axis = np.linspace(0, last_fft / last_sps, last_fft, endpoint=False)
                        if t0 is not None:
                            line_time["ch0"].set_data(t_axis, t0)
                            updated_time = True
                        if t1 is not None:
                            line_time["ch1"].set_data(t_axis, t1)
                            updated_time = True
                        if updated_time:
                            # Auto-scale Y axis based on current frame data
                            y_min, y_max = float('inf'), float('-inf')
                            if t0 is not None:
                                y_min = min(y_min, np.min(t0))
                                y_max = max(y_max, np.max(t0))
                            if t1 is not None:
                                y_min = min(y_min, np.min(t1))
                                y_max = max(y_max, np.max(t1))
                            
                            if y_min != float('inf'):
                                # Calculate center and span of the signal
                                y_span = y_max - y_min
                                y_mid = (y_max + y_min) / 2.0
                                
                                # Default to 1.0 span if DC (flatline)
                                if y_span == 0:
                                    y_span = 1.0
                                else:
                                    # Add small margin (10%) to base auto-scale
                                    y_span *= 1.1

                                # Apply user zoom factor (amp_scale < 1.0 means zoom in)
                                visible_half_span = (y_span * plot_meta.get("amp_scale", 1.0)) / 2.0
                                
                                ax_time.set_ylim(y_mid - visible_half_span, y_mid + visible_half_span)
                                ax_time.set_xlim(0, (last_fft / last_sps) * plot_meta.get("zoom", 1.0))

                    # Update Freq Plot
                    if ax_freq and line_freq["ch0"] is not None and line_freq["ch1"] is not None:
                        # f_axis has length N/2
                        bins_c = len(f0) if f0 is not None else (len(f1) if f1 is not None else 0)
                        if bins_c > 0:
                            f_axis = np.linspace(0, last_sps / 2, bins_c, endpoint=False)
                            
                            # Apply max freq cutoff
                            max_f = float(args.max_freq) if hasattr(args, 'max_freq') else 0.0
                            limit_idx = bins_c
                            if max_f > 0:
                                limit_idx = int(max_f / (last_sps / 2) * bins_c) + 1
                                if limit_idx > bins_c: limit_idx = bins_c
                                
                            f_axis = f_axis[:limit_idx]
                            
                            if f0 is not None:
                                line_freq["ch0"].set_data(f_axis, f0[:limit_idx])
                                updated_freq = True
                            if f1 is not None:
                                line_freq["ch1"].set_data(f_axis, f1[:limit_idx])
                                updated_freq = True
                                
                            if updated_freq:
                                ax_freq.relim()
                                ax_freq.autoscale_view()
                                ax_freq.set_xlim(0, f_axis[-1])
                                ax_freq.set_ylim(-120, 10) # Typical dB range

                for f in figs[1:]:
                    f.canvas.draw_idle()

                return lines

            def update_dummy(_):
                # Just keep the animation loop alive to poll serial
                update(None)
                return lines

            # Attach animation to the first figure to drive the data collection
            # Secondary figures just redraw on idle
            ani = animation.FuncAnimation(
                figs[0],
                update_dummy,
                interval=args.plot_interval,
                blit=False,
                cache_frame_data=False,
            )
            plt.show()
        else:
            # Non-plotting mode: just consume the queue
            while True:
                try:
                    version, seq, payload = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    if stop_requested:
                        raise KeyboardInterrupt
                    continue
                
                row = payload_to_row(version, payload)
                if row is None:
                    continue
                handle_row(seq, row)
                if args.audio and audio_synth is not None:
                    label = str(row[3])
                    if args.audio_channel in label:
                        payload_bytes = row[4]
                        if isinstance(payload_bytes, (bytes, bytearray, memoryview)):
                            bins_list = list(struct.unpack("<%df" % (len(payload_bytes)//4), payload_bytes))
                        else:
                            bins_list = list(payload_bytes)
                        audio_synth.update(bins_list, int(row[1]), int(row[2]))
                if stop_requested:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        duration = time.time() - start
        rate = count / duration if duration else 0.0
        print(f"Captured {count} frames in {duration:.2f}s -> {rate:.1f} frames/s, dropped={dropped}")
        return 0
    except serial.SerialException as exc:
        print(f"Serial error opening {args.port}: {exc}", file=sys.stderr)
        return 1
    finally:
        reader_stop_event.set()
        if file_handle is not None:
            file_handle.close()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
