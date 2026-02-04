import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";

export const About = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const features = [
    {
      title: "CT Imaging Analysis",
      description:
        "Advanced convolutional networks extract spatial features from abdominal CT scans with high precision.",
    },
    {
      title: "Molecular Integration",
      description:
        "Gene expression and mutation profiles are fused with imaging data for comprehensive subtyping.",
    },
    {
      title: "Explainable Results",
      description:
        "Transparent AI provides clinicians with interpretable insights to support decision-making.",
    },
  ];

  return (
    <section id="about" className="py-32 bg-muted/30" ref={ref}>
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left column */}
          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={isInView ? { opacity: 1, x: 0 } : {}}
            transition={{ duration: 0.8 }}
          >
            <span className="text-primary font-medium text-sm uppercase tracking-widest mb-4 block">
              About PanOptic
            </span>
            <h2 className="text-4xl md:text-5xl font-serif font-medium leading-tight mb-6">
              Where imaging meets
              <br />
              <span className="italic">molecular insight.</span>
            </h2>
            <p className="text-muted-foreground text-lg leading-relaxed mb-8">
              Pancreatic cancer remains one of the most challenging malignancies
              to diagnose and treat. PanOptic represents a new paradigm in
              precision oncology — combining the spatial richness of CT imaging
              with the biological depth of molecular profiling.
            </p>
            <p className="text-muted-foreground leading-relaxed">
              Our multimodal approach enables accurate classification of cancer
              subtypes, informing personalized treatment strategies and
              improving patient outcomes through early, precise diagnosis.
            </p>
          </motion.div>

          {/* Right column - Features */}
          <div className="space-y-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                animate={isInView ? { opacity: 1, y: 0 } : {}}
                transition={{ duration: 0.6, delay: 0.2 + index * 0.15 }}
                className="bg-card rounded-2xl p-8 shadow-soft hover:shadow-elevated transition-shadow duration-300"
              >
                <div className="flex items-start gap-4">
                  <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center flex-shrink-0">
                    <span className="text-primary font-serif font-medium">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                  </div>
                  <div>
                    <h3 className="font-serif text-xl font-medium mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};
