import { motion, AnimatePresence } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef, useState } from "react";
import { Upload, FileText, Dna, AlertCircle } from "lucide-react";
import { AnalysisLoader } from "./AnalysisLoader";
import { ResultsDisplay } from "./ResultsDisplay";
import { useToast } from "@/hooks/use-toast";

type AnalysisState = "idle" | "uploading" | "analyzing" | "complete" | "error";

export interface ExplanationSection {
  heading: string;
  body: string;
  highlight?: string;
  breakdown?: string;
  similar_cases?: { subtype: string; similarity: string; matches_prediction: boolean }[];
  alternatives?: { name: string; percentage: number }[];
}

export interface AnalysisResult {
  subtype: string;
  probability: number;
  simpleExplanation: string;
  detailedExplanation: string;
  /** Human-friendly structured sections for detailed view (from API) */
  explanationSections?: ExplanationSection[];
}

const CT_FORMATS = [".nii", ".nii.gz", ".dcm", ".dicom"];
const MOLECULAR_FORMATS = [".csv", ".tsv", ".txt", ".vcf"];
const API_BASE = "https://panoptic-render-1.onrender.com";

// Extract patient ID from filename (assumes format like "patient_123_scan.nii" or "123_data.csv")
const extractPatientId = (filename: string): string | null => {
  // Remove file extension
  const nameWithoutExt = filename.replace(/\.[^/.]+$/, "").replace(/\.nii$/, "");
  
  // Try to find patient ID patterns (numbers, or alphanumeric IDs)
  // Pattern 1: "patient_XXX" or "pt_XXX"
  const patientMatch = nameWithoutExt.match(/(?:patient|pt)[_-]?(\w+)/i);
  if (patientMatch) return patientMatch[1].toLowerCase();
  
  // Pattern 2: Leading ID like "123_something" or "ABC123_something"
  const leadingIdMatch = nameWithoutExt.match(/^([a-zA-Z0-9]+)[_-]/);
  if (leadingIdMatch) return leadingIdMatch[1].toLowerCase();
  
  // Pattern 3: Just use the whole filename without extension as ID
  return nameWithoutExt.toLowerCase();
};

const validateFileFormat = (file: File, type: "ct" | "molecular"): boolean => {
  const filename = file.name.toLowerCase();
  const validFormats = type === "ct" ? CT_FORMATS : MOLECULAR_FORMATS;
  return validFormats.some(format => filename.endsWith(format));
};

export const Prototype = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });
  const { toast } = useToast();
  
  const [ctFile, setCtFile] = useState<File | null>(null);
  const [molecularFile, setMolecularFile] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleFileUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    type: "ct" | "molecular"
  ) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate file format
    if (!validateFileFormat(file, type)) {
      const validFormats = type === "ct" ? CT_FORMATS.join(", ") : MOLECULAR_FORMATS.join(", ");
      toast({
        variant: "destructive",
        title: "Invalid File Format",
        description: `Please upload a valid ${type === "ct" ? "CT scan" : "molecular data"} file. Accepted formats: ${validFormats}`,
      });
      e.target.value = "";
      return;
    }

    if (type === "ct") {
      setCtFile(file);
    } else {
      setMolecularFile(file);
    }
  };

  const handleAnalyze = async () => {
    if (!ctFile || !molecularFile) return;

    // Validate patient ID match
    const ctPatientId = extractPatientId(ctFile.name);
    const molecularPatientId = extractPatientId(molecularFile.name);

    if (ctPatientId !== molecularPatientId) {
      toast({
        variant: "destructive",
        title: "Patient ID Mismatch",
        description: `The CT scan (${ctPatientId}) and molecular data (${molecularPatientId}) files appear to be from different patients. Please ensure both files have matching patient IDs in their filenames.`,
      });
      return;
    }

    try {
      setAnalysisState("uploading");

      const formData = new FormData();
      formData.append("ct_file", ctFile);
      formData.append("molecular_file", molecularFile);
      formData.append("explain", "detailed");

      setAnalysisState("analyzing");

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.detail || `API error: ${res.status}`);
      }

      const data = await res.json();

      const predictedSubtype = data.predicted_subtype;
      const confidence = data.confidence;

      const simpleText =
        data?.explanation?.mode === "simple"
          ? data.explanation.summary_text
          : data?.explanation?.summary_text || "No explanation returned.";

      const details = data?.explanation?.details;
      const sections = details?.sections;
      const detailedText = sections
        ? data.explanation.summary_text
        : data?.explanation?.mode === "detailed"
          ? `${data.explanation.summary_text}\n\n${JSON.stringify(details || data.explanation, null, 2)}`
          : JSON.stringify(data.explanation, null, 2);

      setResult({
        subtype: predictedSubtype,
        probability: confidence,
        simpleExplanation: simpleText,
        detailedExplanation: detailedText,
        explanationSections: sections,
      });

      setAnalysisState("complete");
    } catch (e: any) {
      console.error(e);
      setAnalysisState("idle");
      
      // Show user-friendly error toast
      if (e.message?.includes("fetch") || e.message?.includes("network") || e.name === "TypeError") {
        toast({
          variant: "destructive",
          title: "Connection Error",
          description: "Unable to connect to the analysis server. Please check your internet connection and try again.",
        });
      } else {
        toast({
          variant: "destructive",
          title: "Analysis Failed",
          description: e?.message || "An unexpected error occurred during analysis. Please try again.",
        });
      }
    }
  };

  const handleReset = () => {
    setCtFile(null);
    setMolecularFile(null);
    setAnalysisState("idle");
    setResult(null);
  };

  return (
    <section id="prototype" className="py-32" ref={ref}>
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <span className="text-primary font-medium text-sm uppercase tracking-widest mb-4 block">
            Interactive Demo
          </span>
          <h2 className="text-4xl md:text-5xl font-serif font-medium leading-tight max-w-2xl mx-auto mb-6">
            Experience <span className="italic">PanOptic</span> in action.
          </h2>
          <p className="text-muted-foreground text-lg max-w-xl mx-auto">
            Upload a CT scan and molecular data file to receive AI-powered
            subtype classification with detailed explanations.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="max-w-4xl mx-auto"
        >
          <AnimatePresence mode="wait">
            {analysisState === "idle" && (
              <motion.div
                key="upload"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className="bg-card rounded-3xl p-8 md:p-12 shadow-elevated border border-border/50"
              >
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  {/* CT Upload */}
                  <label
                    className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer ${
                      ctFile
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50 hover:bg-muted/50"
                    }`}
                  >
                    <input
                      type="file"
                      accept=".nii,.nii.gz,.dcm,.dicom"
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      onChange={(e) => handleFileUpload(e, "ct")}
                    />
                    <Upload className="w-10 h-10 text-primary mb-4" />
                    <span className="font-medium text-lg mb-1">
                      {ctFile ? ctFile.name : "CT Scan"}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {ctFile ? "File uploaded" : ".nii, .dcm formats"}
                    </span>
                    {ctFile && (
                      <div className="absolute top-4 right-4 w-6 h-6 rounded-full bg-primary flex items-center justify-center">
                        <svg
                          className="w-4 h-4 text-primary-foreground"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                    )}
                  </label>

                  {/* Molecular Upload */}
                  <label
                    className={`relative flex flex-col items-center justify-center p-8 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer ${
                      molecularFile
                        ? "border-primary bg-primary/5"
                        : "border-border hover:border-primary/50 hover:bg-muted/50"
                    }`}
                  >
                    <input
                      type="file"
                      accept=".csv,.tsv,.txt,.vcf"
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                      onChange={(e) => handleFileUpload(e, "molecular")}
                    />
                    <Dna className="w-10 h-10 text-primary mb-4" />
                    <span className="font-medium text-lg mb-1">
                      {molecularFile ? molecularFile.name : "Molecular Data"}
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {molecularFile ? "File uploaded" : ".csv, .vcf formats"}
                    </span>
                    {molecularFile && (
                      <div className="absolute top-4 right-4 w-6 h-6 rounded-full bg-primary flex items-center justify-center">
                        <svg
                          className="w-4 h-4 text-primary-foreground"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                    )}
                  </label>
                </div>

                {/* File info */}
                {(ctFile || molecularFile) && (
                  <div className="flex flex-wrap gap-4 mb-8 justify-center">
                    {ctFile && (
                      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted text-sm">
                        <FileText className="w-4 h-4 text-primary" />
                        <span className="truncate max-w-[150px]">{ctFile.name}</span>
                      </div>
                    )}
                    {molecularFile && (
                      <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted text-sm">
                        <Dna className="w-4 h-4 text-primary" />
                        <span className="truncate max-w-[150px]">{molecularFile.name}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Analyze button */}
                <div className="text-center">
                  <button
                    onClick={handleAnalyze}
                    disabled={!ctFile || !molecularFile}
                    className={`inline-flex items-center gap-3 px-10 py-4 rounded-full font-medium transition-all duration-300 ${
                      ctFile && molecularFile
                        ? "bg-primary text-primary-foreground hover:shadow-glow cursor-pointer"
                        : "bg-muted text-muted-foreground cursor-not-allowed"
                    }`}
                  >
                    Begin Analysis
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                    >
                      <path
                        d="M3 8h10m0 0L9 4m4 4L9 12"
                        stroke="currentColor"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                </div>

                {/* Disclaimer */}
                <div className="mt-8 flex items-start gap-3 p-4 rounded-xl bg-muted/50 text-sm text-muted-foreground">
                  <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
                  <p>
                    This is a research prototype. Results are for demonstration
                    purposes only and should not be used for clinical
                    decision-making.
                  </p>
                </div>
              </motion.div>
            )}

            {(analysisState === "uploading" || analysisState === "analyzing") && (
              <AnalysisLoader key="loader" state={analysisState} />
            )}

            {analysisState === "complete" && result && (
              <ResultsDisplay
                key="results"
                result={result}
                onReset={handleReset}
              />
            )}
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  );
};
