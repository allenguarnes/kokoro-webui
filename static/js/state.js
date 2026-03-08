export const appState = {
  currentObjectUrl: null,
  playbackToken: 0,
  chunkState: null,
  streamAbortController: null,
  activeSocket: null,
  currentTheme: "dark",
  customSelectRegistry: new Map(),
  lastExportRequest: null,
  availableFormats: ["wav", "opus", "pcm"],
  selectedFormat: "wav",
  availableOpusBitrates: ["16k", "24k", "32k", "48k"],
  availableWavSampleRates: [
    "native",
    "16000",
    "22050",
    "24000",
    "44100",
    "48000",
  ],
  selectedOpusBitrate: "32k",
  selectedWavSampleRate: "native",
  pitchShiftingAvailable: true,
  maxPitchSemitones: 6,
};

export function createChunkState() {
  return {
    queue: [],
    queueWaiter: null,
    pendingChunkMeta: null,
    totalChunks: 0,
    playedChunks: 0,
    totalElapsedMs: 0,
    totalBytes: 0,
    totalDurationSec: 0,
    streamDone: false,
    streamError: null,
  };
}
