import { ApiResponse } from "@/types/types";
import PredictionsCard from "./card/prediction-card";
import AudioVisualizationCards from "./card/audio-visualisation-card";
import FeatureMapsCard from "./card/feature-maps-card";

interface ResultsDisplayProps {
  vizData: ApiResponse;
  isLoading: boolean;
}

export default function ResultsDisplay({ vizData }: ResultsDisplayProps) {
  return (
    <div className="space-y-8">
      <PredictionsCard predictions={vizData.predictions} />

      <AudioVisualizationCards
        inputSpectrogram={vizData.input_spectrogram}
        waveform={vizData.waveform}
      />

      <FeatureMapsCard visualization={vizData.visualization} />
    </div>
  );
}
