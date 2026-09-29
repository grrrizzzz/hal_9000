import { BarChart3, Gavel, Layers, ListChecks } from "lucide-react";
import { useSearchParams } from "react-router";
import { GoldReview } from "@/components/eval/GoldReview";
import { JudgeCalibration } from "@/components/eval/JudgeCalibration";
import { ProfilePanel } from "@/components/eval/ProfilePanel";
import { SweepRuns } from "@/components/eval/SweepRuns";
import { Page, PageHeader } from "@/components/layout/PageHeader";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const TABS = ["review", "judge", "runs", "profile"] as const;
type Tab = (typeof TABS)[number];

export default function EvaluationPage() {
  const [params, setParams] = useSearchParams();
  const tab: Tab = (TABS as readonly string[]).includes(params.get("tab") ?? "") ? (params.get("tab") as Tab) : "review";
  return (
    <Page wide>
      <PageHeader title="Evaluation" description="Review the question set, calibrate the judge, compare sweep runs and choose the index profile." />
      <Tabs
        value={tab}
        onValueChange={(v) => setParams({ tab: v }, { replace: true })}
        className="mt-6"
      >
        <TabsList aria-label="Evaluation sections">
          <TabsTrigger value="review">
            <ListChecks /> Review queue
          </TabsTrigger>
          <TabsTrigger value="judge">
            <Gavel /> Judge calibration
          </TabsTrigger>
          <TabsTrigger value="runs">
            <BarChart3 /> Sweep runs
          </TabsTrigger>
          <TabsTrigger value="profile">
            <Layers /> Index profile
          </TabsTrigger>
        </TabsList>
        <TabsContent value="review" className="mt-5">
          <GoldReview />
        </TabsContent>
        <TabsContent value="judge" className="mt-5">
          <JudgeCalibration />
        </TabsContent>
        <TabsContent value="runs" className="mt-5">
          <SweepRuns />
        </TabsContent>
        <TabsContent value="profile" className="mt-5">
          <ProfilePanel />
        </TabsContent>
      </Tabs>
    </Page>
  );
}
