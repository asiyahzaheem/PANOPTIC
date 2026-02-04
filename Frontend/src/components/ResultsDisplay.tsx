import { motion } from "framer-motion";
import { useState } from "react";
import { RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import type { AnalysisResult } from "./Prototype";

interface ResultsDisplayProps {
  result: AnalysisResult;
  onReset: () => void;
}

export const ResultsDisplay = ({ result, onReset }: ResultsDisplayProps) => {
  const [showDetailed, setShowDetailed] = useState(false);

  const probabilityPercentage = Math.round(result.probability * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-card rounded-3xl p-8 md:p-12 shadow-elevated border border-border/50"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-8">
        <div>
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-100 text-green-700 text-sm font-medium mb-4">
            <span className="w-2 h-2 rounded-full bg-green-500" />
            Analysis Complete
          </span>
          <h3 className="font-serif text-2xl md:text-3xl font-medium">
            Classification Results
          </h3>
        </div>
        <button
          onClick={onReset}
          className="flex items-center gap-2 px-4 py-2 rounded-full border border-border hover:bg-muted transition-colors text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          New Analysis
        </button>
      </div>

      {/* Main result */}
      <div className="grid md:grid-cols-2 gap-8 mb-8">
        {/* Subtype */}
        <div className="bg-muted/50 rounded-2xl p-6">
          <span className="text-sm text-muted-foreground uppercase tracking-wide mb-2 block">
            Predicted Subtype
          </span>
          <h4 className="font-serif text-2xl font-medium text-primary mb-2">
            {result.subtype}
          </h4>
        </div>

        {/* Confidence */}
        <div className="bg-muted/50 rounded-2xl p-6">
          <span className="text-sm text-muted-foreground uppercase tracking-wide mb-2 block">
            Confidence Score
          </span>
          <div className="flex items-end gap-2">
            <span className="font-serif text-4xl font-medium text-primary">
              {probabilityPercentage}%
            </span>
            <span className="text-muted-foreground mb-1">confidence</span>
          </div>
          <div className="mt-3 h-2 rounded-full bg-muted overflow-hidden">
            <motion.div
              className="h-full bg-primary rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${probabilityPercentage}%` }}
              transition={{ duration: 1, delay: 0.5 }}
            />
          </div>
        </div>
      </div>

      {/* Explanation toggle */}
      <div className="border-t border-border pt-8">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-serif text-xl font-medium">Explanation</h4>
          <button
            onClick={() => setShowDetailed(!showDetailed)}
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted hover:bg-secondary transition-colors text-sm font-medium"
          >
            {showDetailed ? "Simple View" : "Detailed View"}
            {showDetailed ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        </div>

        <motion.div
          key={showDetailed ? "detailed" : "simple"}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-muted/30 rounded-2xl p-6"
        >
          {showDetailed ? (
            <div className="prose prose-sm max-w-none">
              <div className="whitespace-pre-wrap text-muted-foreground leading-relaxed">
                {result.detailedExplanation.split("\n").map((line, index) => {
                  if (line.startsWith("**") && line.endsWith("**")) {
                    return (
                      <h5
                        key={index}
                        className="font-medium text-foreground mt-4 mb-2 first:mt-0"
                      >
                        {line.replace(/\*\*/g, "")}
                      </h5>
                    );
                  }
                  if (line.startsWith("- ")) {
                    return (
                      <div key={index} className="flex gap-2 ml-2 my-1">
                        <span className="text-primary">•</span>
                        <span>{line.substring(2)}</span>
                      </div>
                    );
                  }
                  return line ? (
                    <p key={index} className="my-1">
                      {line}
                    </p>
                  ) : (
                    <br key={index} />
                  );
                })}
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground leading-relaxed">
              {result.simpleExplanation}
            </p>
          )}
        </motion.div>
      </div>

      {/* Disclaimer */}
      <div className="mt-8 p-4 rounded-xl bg-secondary/50 text-sm text-secondary-foreground">
        <strong>Research Use Only:</strong> These results are generated by an AI
        model and should be validated by qualified medical professionals before
        any clinical application.
      </div>
    </motion.div>
  );
};
