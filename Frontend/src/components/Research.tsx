import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";

export const Research = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const methods = [
    {
      title: "Vision Transformer Backbone",
      content:
        "Our imaging pipeline uses a Vision Transformer (ViT) architecture, pre-trained on medical imaging datasets and fine-tuned for pancreatic lesion feature extraction.",
    },
    {
      title: "Graph Neural Networks",
      content:
        "Molecular data is processed through graph neural networks that capture gene-gene interactions and pathway-level relationships.",
    },
    {
      title: "Cross-Modal Attention",
      content:
        "A novel attention mechanism aligns imaging features with molecular signatures, enabling the model to learn complementary representations.",
    },
    {
      title: "Ensemble Prediction",
      content:
        "Final classification leverages ensemble methods across multiple modalities, with calibrated confidence scores for clinical utility.",
    },
  ];

  return (
    <section id="research" className="py-32 bg-muted/30" ref={ref}>
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="max-w-3xl mb-16"
        >
          <span className="text-primary font-medium text-sm uppercase tracking-widest mb-4 block">
            Research & Methods
          </span>
          <h2 className="text-4xl md:text-5xl font-serif font-medium leading-tight mb-6">
            Built on rigorous
            <br />
            <span className="italic">scientific foundations.</span>
          </h2>
          <p className="text-muted-foreground text-lg leading-relaxed">
            PanOptic integrates state-of-the-art deep learning architectures
            with domain expertise in oncology and radiology. Our approach is
            validated on large-scale clinical datasets.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          {methods.map((method, index) => (
            <motion.div
              key={method.title}
              initial={{ opacity: 0, y: 30 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: 0.2 + index * 0.1 }}
              className="relative"
            >
              <div className="flex gap-6">
                <div className="flex-shrink-0">
                  <div className="w-12 h-12 rounded-full border-2 border-primary/30 flex items-center justify-center">
                    <span className="text-primary font-serif font-medium text-sm">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                  </div>
                  {index < methods.length - 1 && (
                    <div className="w-px h-full bg-border mx-auto mt-4 hidden md:block" />
                  )}
                </div>
                <div className="pb-8">
                  <h3 className="font-serif text-xl font-medium mb-3">
                    {method.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed">
                    {method.content}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Tech stack visual */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="mt-16 p-8 rounded-2xl bg-card border border-border/50"
        >
          <div className="flex flex-wrap justify-center gap-8 text-sm text-muted-foreground">
            {[
              "PyTorch",
              "Transformers",
              "MONAI",
              "scikit-learn",
              "NumPy",
              "Pandas",
            ].map((tech) => (
              <span
                key={tech}
                className="px-4 py-2 rounded-full bg-muted hover:bg-secondary transition-colors duration-300"
              >
                {tech}
              </span>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
};
