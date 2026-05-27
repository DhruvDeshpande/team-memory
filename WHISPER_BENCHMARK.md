# Whisper Benchmark Notes

Benchmark video: `videos/final_test.mp4`  
Video duration: ~57 seconds  
Frame interval: 5 seconds  
Hardware: Local MacBook  
Backend: Faster Whisper

## Results

| Model | Transcription Time | Total Pipeline Time | Notes |
|---|---:|---:|---|
| tiny | 13.61s | 14.60s | Slower than expected in this run |
| base | 6.14s | 7.07s | Best current default |
| small | 44.47s | 45.35s | Much slower, may improve quality but not ideal for quick local processing |

## Initial Conclusion

For the current Phase 1 prototype, `base` is the best default model because it gives the fastest observed total pipeline time while still producing usable transcripts.

For longer executive meeting recordings, estimated processing time with `base` is roughly:

57s video → ~7s processing  
1 hour video → ~7–8 minutes processing

This estimate may vary based on meeting audio quality, hardware, and whether OCR or slide-change detection is added later.

## Next Improvements

- Test on a real 30–60 minute meeting recording.
- Compare transcript quality, not just speed.
- Add optional `medium` benchmark if quality is significantly better.
- Explore Apple Silicon optimized options like `mlx-whisper`.