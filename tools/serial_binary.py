#!/usr/bin/env python3
import argparse
import binascii
import csv
import struct
import sys
import time
import threading
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
    data = struct.unpack("<%df" % (len(data_bytes) // 4), data_bytes)
    
    if version == 1:
        if fft and len(data) != fft // 2:
            return None
    elif version == 2:
        if fft and len(data) != fft:
            return None
    else:
        # Unknown version
        pass
        
    return [t_ms, sps, fft, label_text, *data]


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
    parser.add_argument("--audio", action="store_true", default=False, help="Play audio representation of FFT magnitudes")
    parser.add_argument("--audio-rate", type=int, default=44100, help="Audio sample rate in Hz")
    parser.add_argument("--audio-block", type=int, default=1024, help="Audio block size in samples")
    parser.add_argument("--audio-gain", type=float, default=0.0, help="Audio gain scaling (0.0-1.0)")
    parser.add_argument("--audio-channel", choices=["ch0", "ch1"], default="ch0", help="Channel to sonify")
    parser.add_argument("--no-csv", action="store_true", default=False, help="Disable CSV output (plot only)")
    parser.add_argument("--reconnect", action="store_true", default=True, help="Auto-reconnect on device reset")
    parser.add_argument("--reconnect-delay", type=float, default=0.5, help="Seconds between reconnect attempts")
    parser.add_argument("--no-plot-freq", action="store_true", help="Disable Frequency Domain plot within plotting mode")
    parser.add_argument("--no-plot-time", action="store_true", help="Disable Time Domain plot within plotting mode")
    parser.add_argument("-w", "--window", choices=['0', '1', '2', '3'], default='3', help="Windowing: 0=None, 1=Hann, 2=Hamming, 3=Blackman")
    parser.add_argument("--inspect-frames", action="store_true", help="Print frame metadata and payload sample for debugging")
    args = parser.parse_args()

    mode = "a" if args.append else "w"
    buffer = bytearray()
    last_seq = None
    dropped = 0
    count = 0
    start = time.time()

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
            def __init__(self, rate: int, gain: float, max_freq: float):
                self.rate = rate
                self.gain = gain
                self.max_freq = max_freq
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
                    mags = np.power(10.0, mags / 20.0)

                n_fft = frames * 2
                out_bins = n_fft // 2 + 1
                freqs_in = np.linspace(0.0, sps / 2.0, mags.size)
                freqs_out = np.linspace(0.0, self.rate / 2.0, out_bins)
                mags_out = np.interp(freqs_out, freqs_in, mags, left=0.0, right=0.0)

                if self.max_freq and self.max_freq > 0.0:
                    mags_out[freqs_out > self.max_freq] = 0.0

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
                outdata[:, 0] = audio.astype(np.float32)

        audio_synth = AudioSynth(args.audio_rate, args.audio_gain, args.max_freq)
        audio_stream = sd.OutputStream(
            samplerate=args.audio_rate,
            blocksize=args.audio_block,
            channels=1,
            callback=audio_synth.callback,
        )
        audio_stream.start()

    def handle_row(seq: int, row: list):
        nonlocal last_seq, dropped, count, stop_requested
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

    try:
        if not args.no_csv:
            file_handle = args.output.open(mode, newline="")
            assert file_handle is not None
            writer = csv.writer(file_handle)

        def open_serial_port() -> Optional[serial.Serial]:
            try:
                ser = serial.Serial(args.port, args.baud, timeout=0.1)
                print(f"Connected to {args.port}")
                return ser
            except serial.SerialException as exc:
                if args.reconnect:
                    print(f"Waiting for {args.port}: {exc}", file=sys.stderr)
                    return None
                raise

        if args.plot:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
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

            if not args.no_plot_time:
                fig_time = plt.figure("Time Domain", figsize=(10, 5))
                figs.append(fig_time)
                ax_time = fig_time.add_subplot(111)
                l0_t, = ax_time.plot([], [], label="ch0")
                l1_t, = ax_time.plot([], [], label="ch1")
                lines.extend([l0_t, l1_t])
                ax_time.set_xlabel("Time (s)")
                ax_time.set_ylabel("Amplitude")
                ax_time.legend(loc="upper right")
                ax_time.grid(True)

            if not args.no_plot_freq:
                fig_freq = plt.figure("Frequency Domain", figsize=(10, 5))
                figs.append(fig_freq)
                ax_freq = fig_freq.add_subplot(111)
                l0_f, = ax_freq.plot([], [], label="ch0")
                l1_f, = ax_freq.plot([], [], label="ch1")
                lines.extend([l0_f, l1_f])
                ax_freq.set_xlabel("Frequency (Hz)")
                ax_freq.set_ylabel("Magnitude (dB)")
                ax_freq.legend(loc="upper right")
                ax_freq.grid(True)
            
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
                nonlocal ser, next_reconnect, buffer, last_seq
                if ser is None:
                    if args.reconnect and time.time() >= next_reconnect:
                        ser = open_serial_port()
                        next_reconnect = time.time() + args.reconnect_delay
                        if ser is not None:
                            buffer.clear()
                            last_seq = None
                    return lines
                try:
                    # Read all available bytes, or at least 4096 to prevent blocking small reads
                    waiting = ser.in_waiting
                    read_size = waiting if waiting > 4096 else 4096
                    chunk = ser.read(read_size)
                except serial.SerialException as exc:
                    print(f"Serial error while reading {args.port}: {exc}", file=sys.stderr)
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    next_reconnect = time.time() + args.reconnect_delay
                    return lines
                if chunk:
                    buffer.extend(chunk)
                frames = parse_frames(buffer)
                for version, seq, payload in frames:
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
                    bins = row[4:]
                    plot_meta["sps"] = int(row[1])
                    plot_meta["fft"] = int(row[2])
                    
                    # Convert list to array for storage
                    if "ch0" in label:
                        last_bins["ch0"] = bins
                    elif "ch1" in label:
                        last_bins["ch1"] = bins
                        
                    if args.audio and audio_synth is not None:
                        if args.audio_channel in label:
                            audio_synth.update(bins, plot_meta["sps"], plot_meta["fft"])

                # Update Plots
                updated_time = False
                updated_freq = False
                
                last_sps = plot_meta["sps"]
                last_fft = plot_meta["fft"]
                
                if last_sps and last_fft and (last_bins["ch0"] is not None or last_bins["ch1"] is not None):
                    # Helper to process one channel
                    def process_channel(bins_list):
                        if bins_list is None: return None, None
                        raw_in = np.array(bins_list)
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
                    if ax_time:
                        t_axis = np.linspace(0, last_fft / last_sps, last_fft, endpoint=False)
                        if t0 is not None:
                            lines[0].set_data(t_axis, t0)
                            updated_time = True
                        if t1 is not None:
                            lines[1].set_data(t_axis, t1)
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
                    if ax_freq:
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
                                lines[-2].set_data(f_axis, f0[:limit_idx])
                                updated_freq = True
                            if f1 is not None:
                                lines[-1].set_data(f_axis, f1[:limit_idx])
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
            ser = open_serial_port()
            next_reconnect = time.time()
            while True:
                if ser is None:
                    if args.reconnect and time.time() >= next_reconnect:
                        ser = open_serial_port()
                        next_reconnect = time.time() + args.reconnect_delay
                        if ser is not None:
                            buffer.clear()
                            last_seq = None
                    time.sleep(0.05)
                    continue
                try:
                    # Read all available bytes, or at least 4096 to prevent blocking small reads
                    waiting = ser.in_waiting
                    read_size = waiting if waiting > 4096 else 4096
                    chunk = ser.read(read_size)
                except serial.SerialException as exc:
                    print(f"Serial error while reading {args.port}: {exc}", file=sys.stderr)
                    try:
                        ser.close()
                    except Exception:
                        pass
                    ser = None
                    next_reconnect = time.time() + args.reconnect_delay
                    continue
                if chunk:
                    buffer.extend(chunk)
                frames = parse_frames(buffer)
                for version, seq, payload in frames:
                    row = payload_to_row(version, payload)
                    if row is None:
                        continue
                    handle_row(seq, row)
                    if args.audio and audio_synth is not None:
                        label = str(row[3])
                        if args.audio_channel in label:
                            audio_synth.update(row[4:], int(row[1]), int(row[2]))
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
        if file_handle is not None:
            file_handle.close()
        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
