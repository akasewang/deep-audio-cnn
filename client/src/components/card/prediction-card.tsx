import { Prediction } from "@/types/types";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import { getEmojiForClass } from "@/lib/utils";
import { Badge } from "../ui/badge";
import { Progress } from "../ui/progress";

interface PredictionsCardProps {
  predictions: Prediction[];
}

export default function PredictionsCard({ predictions }: PredictionsCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-stone-900">Top Predictions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {predictions.slice(0, 3).map((pred, i) => (
            <div key={pred.class} className="space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-md font-medium text-stone-700">
                  {getEmojiForClass(pred.class)}{" "}
                  <span>{pred.class.replaceAll("_", " ")}</span>
                </div>
                <Badge variant={i === 0 ? "default" : "secondary"}>
                  {(pred.confidence * 100).toFixed(1)}%
                </Badge>
              </div>
              <Progress value={pred.confidence * 100} className="h-2" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
