import { FolderOpen, FilePlus2, UploadCloud } from "lucide-react";
import { useRef } from "react";
import { useDropzone, type DropEvent } from "react-dropzone";
import { Button } from "@/components/ui/button";
import { filesFromDataTransfer, filesFromInput, type PickedFile } from "@/lib/upload/walker";
import { cn } from "@/lib/utils";

const PATH = Symbol("relativePath");
type Tagged = File & { [PATH]?: string };

/** react-dropzone hands back Files; we carry each file's relative path on it. */
async function getFilesFromEvent(event: DropEvent): Promise<(File | DataTransferItem)[]> {
  const e = event as DragEvent & { target: HTMLInputElement };
  if (e.type === "drop" && e.dataTransfer) {
    const picked = await filesFromDataTransfer(e.dataTransfer);
    return picked.map(tag);
  }
  if (e.dataTransfer) {
    // dragenter/dragover: only item kinds are readable here.
    return Array.from(e.dataTransfer.items ?? []).filter((i) => i.kind === "file");
  }
  if (e.target && "files" in e.target) return filesFromInput(e.target.files).map(tag);
  return [];
}

function tag(p: PickedFile): File {
  const f = p.file as Tagged;
  f[PATH] = p.relativePath;
  return f;
}

const untag = (f: File): PickedFile => ({ file: f, relativePath: (f as Tagged)[PATH] ?? f.name });

export function DropZone({ onPick, disabled }: { onPick: (files: PickedFile[]) => void; disabled?: boolean }) {
  const folderInput = useRef<HTMLInputElement>(null);
  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    noClick: true,
    noKeyboard: true,
    multiple: true,
    disabled,
    useFsAccessApi: false,
    getFilesFromEvent,
    onDrop: (accepted) => onPick(accepted.map(untag)),
  });

  return (
    <div
      {...getRootProps({
        className: cn(
          "relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed border-border-strong bg-surface px-6 py-10 text-center transition-colors sm:py-14",
          isDragActive && "border-accent bg-accent-soft",
          disabled && "opacity-60",
        ),
        "aria-label": "Drop zone for files and folders",
        "data-testid": "drop-zone",
      } as Parameters<typeof getRootProps>[0])}
    >
      <input {...getInputProps({ "data-testid": "files-input" } as Parameters<typeof getInputProps>[0])} />
      <input
        ref={folderInput}
        type="file"
        className="hidden"
        data-testid="folder-input"
        multiple
        // Non-standard but supported by every current browser.
        {...({ webkitdirectory: "", directory: "" } as Record<string, string>)}
        onChange={(e) => {
          onPick(filesFromInput(e.target.files));
          e.target.value = "";
        }}
      />
      <div className="grid size-12 place-items-center rounded-2xl bg-accent-soft text-accent-foreground">
        <UploadCloud className="size-6" aria-hidden />
      </div>
      <p className="mt-4 text-[15px] font-medium">{isDragActive ? "Drop to add" : "Drop files or folders here"}</p>
      <p className="mt-1 text-sm text-muted-foreground">PDF, Word (.docx), HTML, Markdown and text · up to 100 MB each</p>
      <div className="mt-5 flex flex-wrap justify-center gap-2">
        <Button onClick={open} disabled={disabled}>
          <FilePlus2 /> Choose files
        </Button>
        <Button onClick={() => folderInput.current?.click()} disabled={disabled}>
          <FolderOpen /> Choose folder
        </Button>
      </div>
    </div>
  );
}
