import { Compass } from "lucide-react";
import { Link } from "react-router";
import { Button } from "@/components/ui/button";
import { EmptyState } from "@/components/ui/states";

export function NotFoundPage() {
  return (
    <EmptyState icon={<Compass />} title="Nothing here" className="py-24" action={<Button asChild><Link to="/">Back to chat</Link></Button>}>
      That page doesn't exist.
    </EmptyState>
  );
}
