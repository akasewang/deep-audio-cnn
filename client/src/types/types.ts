export interface Prediction {
  class: string;
  confidence: number;
}

export interface LayerData {
  shape: number[];
  values: number[][];
}

export interface VisualizationData {
  [layerName: string]: LayerData;
}

export interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

export interface ApiResponse {
  predictions: Prediction[];
  visualization: VisualizationData;
  input_spectrogram: LayerData;
  waveform: WaveformData;
}
