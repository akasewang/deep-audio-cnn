import { Badge } from "./ui/badge";
import { Button } from "./ui/button";

interface FileUploaderProps {
  onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  isLoading: boolean;
  fileName: string;
}

export default function FileUploader({
  onFileChange,
  isLoading,
  fileName,
}: FileUploaderProps) {
  return (
    <div className="mb-12 text-center">
      <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
        CNN Audio Visualizer
      </h1>
      <p className="text-md mb-8 text-stone-600">
        Upload a WAV file to see the model&apos;s predictions and feature maps
      </p>

      <div className="flex flex-col items-center">
        <div className="relative inline-block">
          <input
            type="file"
            accept=".wav"
            id="file-upload"
            onChange={onFileChange}
            disabled={isLoading}
            className="absolute inset-0 w-full cursor-pointer opacity-0"
          />
          <Button
            disabled={isLoading}
            className="border-stone-300"
            variant="outline"
            size="lg"
          >
            {isLoading ? "Analysing..." : "Choose File"}
          </Button>
        </div>

        {fileName && (
          <Badge
            variant="secondary"
            className="mt-4 bg-stone-200 text-stone-700"
          >
            {fileName}
          </Badge>
        )}
      </div>
    </div>
  );
}
