import { Card, CardContent } from "./ui/card";

interface ErrorDisplayProps {
  error: string;
}

export default function ErrorDisplay({ error }: ErrorDisplayProps) {
  return (
    <Card className="mb-8 border-red-200 bg-red-50">
      <CardContent>
        <p className="text-red-600">Error: {error}</p>
      </CardContent>
    </Card>
  );
}
