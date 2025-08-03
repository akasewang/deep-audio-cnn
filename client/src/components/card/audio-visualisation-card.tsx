import { LayerData, WaveformData } from "@/types/types";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import FeatureMap from "../visualisation/feature-map";
import ColorScale from "../visualisation/color-scale";
import Waveform from "../visualisation/waveform";

interface AudioVisualizationCardsProps {
  inputSpectrogram: LayerData;
  waveform: WaveformData;
}

export default function AudioVisualizationCards({
  inputSpectrogram,
  waveform,
}: AudioVisualizationCardsProps) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      <Card>
        <CardHeader className="text-stone-900">
          <CardTitle className="text-stone-900">Input Spectrogram</CardTitle>
        </CardHeader>
        <CardContent>
          <FeatureMap
            data={inputSpectrogram.values}
            title={`${inputSpectrogram.shape.join(" x ")}`}
            spectrogram
          />
          <div className="mt-5 flex justify-end">
            <ColorScale width={200} height={16} min={-1} max={1} />
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle className="text-stone-900">Audio Waveform</CardTitle>
        </CardHeader>
        <CardContent>
          <Waveform
            data={waveform.values}
            title={`${waveform.duration.toFixed(2)}s * ${
              waveform.sample_rate
            }Hz`}
          />
        </CardContent>
      </Card>
    </div>
  );
}
