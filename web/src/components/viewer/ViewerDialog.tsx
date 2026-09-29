import type { ReactNode } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { DocumentViewer, type ViewerTarget } from "./DocumentViewer";

export function ViewerDialog({
  target,
  title,
  description,
  onClose,
  onUseSelection,
}: {
  target: ViewerTarget | null;
  title: ReactNode;
  description?: ReactNode;
  onClose: () => void;
  onUseSelection?: (quote: string, page: number) => void;
}) {
  return (
    <Dialog open={!!target} onOpenChange={(o) => !o && onClose()}>
      {target ? (
        <DialogContent size="full" title={title} description={description} className="h-[92dvh]">
          <DocumentViewer target={target} onUseSelection={onUseSelection} />
        </DialogContent>
      ) : null}
    </Dialog>
  );
}
