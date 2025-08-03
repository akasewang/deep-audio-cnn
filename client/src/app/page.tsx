"use client";

import ErrorDisplay from "@/components/error-display";
import FileUploader from "@/components/file-uploader";
import MainLayout from "@/components/main-layout";
import ResultsDisplay from "@/components/result-display";
import config from "@/config";
import { ApiResponse } from "@/types/types";
import { useState } from "react";

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            ""
          )
        );

        const response = await fetch(config.apiEndpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_data: base64String }),
        });

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unknown error occured"
        );
      } finally {
        setIsLoading(false);
      }
    };
    reader.onerror = () => {
      setError("Failed ot read the file.");
      setIsLoading(false);
    };
  };

  return (
    <MainLayout>
      <FileUploader
        onFileChange={handleFileChange}
        isLoading={isLoading}
        fileName={fileName}
      />

      {error && <ErrorDisplay error={error} />}

      {vizData && <ResultsDisplay vizData={vizData} isLoading={isLoading} />}
    </MainLayout>
  );
}
