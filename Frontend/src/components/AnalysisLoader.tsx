import { motion } from "framer-motion";
import { useEffect, useState } from "react";

interface AnalysisLoaderProps {
  state: "uploading" | "analyzing";
}

const stages = [
  { label: "Uploading files", duration: 1500 },
  { label: "Preprocessing CT scan", duration: 1000 },
  { label: "Extracting imaging features", duration: 1000 },
  { label: "Processing molecular data", duration: 800 },
  { label: "Running multimodal fusion", duration: 700 },
  { label: "Computing classification", duration: 500 },
];

export const AnalysisLoader = ({ state }: AnalysisLoaderProps) => {
  const [currentStage, setCurrentStage] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (state === "uploading") {
      setCurrentStage(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStage((prev) => {
        if (prev < stages.length - 1) return prev + 1;
        return prev;
      });
    }, 800);

    return () => clearInterval(interval);
  }, [state]);

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev < 100) return prev + 1;
        return prev;
      });
    }, 50);

    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-card rounded-3xl p-8 md:p-12 shadow-elevated border border-border/50"
    >
      <div className="text-center mb-12">
        <h3 className="font-serif text-2xl md:text-3xl font-medium mb-2">
          {state === "uploading" ? "Processing Upload" : "Analyzing Data"}
        </h3>
        <p className="text-muted-foreground">
          Please wait while we process your files
        </p>
      </div>

      {/* Visual loading animation */}
      <div className="relative h-64 mb-12 rounded-2xl bg-muted/30 overflow-hidden">
        {/* Grid background */}
        <div className="absolute inset-0 opacity-30">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern
                id="grid"
                width="20"
                height="20"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 20 0 L 0 0 0 20"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-primary/30"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
          </svg>
        </div>

        {/* Scan line */}
        <motion.div
          className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent"
          animate={{ top: ["0%", "100%", "0%"] }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />

        {/* Central visualization */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative">
            {/* Outer ring */}
            <motion.div
              className="w-32 h-32 rounded-full border-2 border-primary/30"
              animate={{ rotate: 360 }}
              transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Middle ring */}
            <motion.div
              className="absolute inset-4 rounded-full border-2 border-dashed border-primary/50"
              animate={{ rotate: -360 }}
              transition={{ duration: 6, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Inner pulse */}
            <motion.div
              className="absolute inset-8 rounded-full bg-primary/20"
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            
            {/* Center dot */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                className="w-4 h-4 rounded-full bg-primary"
                animate={{ scale: [1, 1.3, 1] }}
                transition={{ duration: 1, repeat: Infinity }}
              />
            </div>
          </div>
        </div>

        {/* Corner decorations */}
        <div className="absolute top-4 left-4 text-xs font-mono text-primary/60">
          PanOptic v1.0
        </div>
        <div className="absolute top-4 right-4 text-xs font-mono text-primary/60">
          {progress}%
        </div>
        <div className="absolute bottom-4 left-4 text-xs font-mono text-primary/60">
          MULTIMODAL
        </div>
        <div className="absolute bottom-4 right-4 text-xs font-mono text-primary/60">
          ANALYSIS
        </div>
      </div>

      {/* Progress stages */}
      <div className="space-y-3">
        {stages.map((stage, index) => (
          <div
            key={stage.label}
            className={`flex items-center gap-3 transition-opacity duration-300 ${
              index <= currentStage ? "opacity-100" : "opacity-30"
            }`}
          >
            <div
              className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-mono ${
                index < currentStage
                  ? "bg-primary text-primary-foreground"
                  : index === currentStage
                  ? "bg-primary/20 text-primary animate-pulse-soft"
                  : "bg-muted text-muted-foreground"
              }`}
            >
              {index < currentStage ? (
                <svg
                  className="w-3 h-3"
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
              ) : (
                String(index + 1).padStart(2, "0")
              )}
            </div>
            <span
              className={`text-sm ${
                index === currentStage
                  ? "text-foreground font-medium"
                  : "text-muted-foreground"
              }`}
            >
              {stage.label}
            </span>
            {index === currentStage && (
              <motion.div
                className="ml-auto flex gap-1"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <motion.div
                  className="w-1.5 h-1.5 rounded-full bg-primary"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0 }}
                />
                <motion.div
                  className="w-1.5 h-1.5 rounded-full bg-primary"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.2 }}
                />
                <motion.div
                  className="w-1.5 h-1.5 rounded-full bg-primary"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1, repeat: Infinity, delay: 0.4 }}
                />
              </motion.div>
            )}
          </div>
        ))}
      </div>

      {/* Bottom progress bar */}
      <div className="mt-8 h-1 rounded-full bg-muted overflow-hidden">
        <motion.div
          className="h-full bg-primary rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.1 }}
        />
      </div>
    </motion.div>
  );
};
