"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";

type AskResponse = {
  answer: string;
  sources: string[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

export default function HomePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string>("");

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string>("");
  const [sources, setSources] = useState<string[]>([]);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string>("");

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const onPickFiles = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const onFilesSelected = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files;
    if (!f) return;
    setFiles(Array.from(f));
  }, []);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(Array.from(e.dataTransfer.files));
      e.dataTransfer.clearData();
    }
  }, []);

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  }, []);

  const upload = useCallback(async () => {
    setError("");
    setUploadMsg("");
    if (files.length === 0) {
      setError("Select files to upload.");
      return;
    }
    try {
      setUploading(true);
      const form = new FormData();
      for (const f of files) form.append("files", f);
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Upload failed (${res.status})`);
      }
      const data = await res.json();
      setUploadMsg(`Uploaded ${data.files?.length ?? 0} file(s), chunks added: ${data.chunks_added ?? 0}`);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setUploading(false);
    }
  }, [files]);

  const ask = useCallback(async () => {
    setError("");
    setAnswer("");
    setSources([]);
    const q = question.trim();
    if (!q) {
      setError("Enter a question.");
      return;
    }
    try {
      setAsking(true);
      const res = await fetch(`${API_BASE}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Ask failed (${res.status})`);
      }
      const data: AskResponse = await res.json();
      setAnswer(data.answer || "");
      setSources(data.sources || []);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setAsking(false);
    }
  }, [question]);

  return (
    <main className="min-h-screen p-6 md:p-10 lg:p-16 bg-white">
      <div className="mx-auto w-full max-w-5xl space-y-8">
        <h1 className="text-xl font-semibold">RAG App (Chroma + Gemini)</h1>

        {error ? (
          <div className="text-sm text-red-600">{error}</div>
        ) : null}

        {/* Upload */}
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-base font-medium">Upload documents</h2>
            <Button variant="secondary" onClick={onPickFiles} disabled={uploading}>
              Choose files
            </Button>
            <input
              ref={fileInputRef}
              type="file"
              className="hidden"
              multiple
              accept=".txt,.md,.pdf"
              onChange={onFilesSelected}
            />
          </div>
          <Separator className="my-3" />

          <div
            className="rounded-md border border-dashed p-6 text-sm text-gray-700"
            onDrop={onDrop}
            onDragOver={onDragOver}
          >
            Drag & drop files here (PDF, TXT, MD)
          </div>

          <div className="mt-3 text-sm text-gray-600">
            {files.length ? `${files.length} file(s) selected` : "No files selected"}
          </div>

          <div className="mt-3 flex gap-2">
            <Button onClick={upload} disabled={uploading}>
              {uploading ? "Uploading..." : "Upload & Ingest"}
            </Button>
            {uploadMsg ? <div className="text-sm text-green-700">{uploadMsg}</div> : null}
          </div>
        </Card>

        {/* Chat */}
        <Card className="p-4">
          <h2 className="text-base font-medium">Ask a question</h2>
          <Separator className="my-3" />
          <div className="grid gap-3">
            <Label htmlFor="q">Question</Label>
            <Input
              id="q"
              placeholder="What is LangChain?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />

            <div className="flex gap-2">
              <Button onClick={ask} disabled={asking}>
                {asking ? "Asking..." : "Ask"}
              </Button>
            </div>

            <div className="grid gap-2">
              <Label>Answer</Label>
              <ScrollArea className="h-40 rounded-md border p-3 text-sm">
                {answer || ""}
              </ScrollArea>
              {sources?.length ? (
                <div className="text-xs text-gray-600">
                  Sources:
                  <ul className="list-disc pl-5">
                    {sources.map((s, i) => (
                      <li key={`${s}-${i}`}>{s}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          </div>
        </Card>
      </div>
    </main>
  );
}
