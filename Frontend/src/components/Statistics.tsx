import { motion } from "framer-motion";
import { useInView } from "framer-motion";
import { useRef } from "react";

export const Statistics = () => {
  const ref = useRef(null);
  const isInView = useInView(ref, { once: true, margin: "-100px" });

  const stats = [
    {
      value: "60,430",
      label: "New Cases (2024)",
      description: "Estimated new pancreatic cancer diagnoses in the US alone",
    },
    {
      value: "13%",
      label: "5-Year Survival",
      description: "One of the lowest survival rates among major cancers",
    },
    {
      value: "4th",
      label: "Leading Cause",
      description: "Of cancer-related deaths in the United States",
    },
    {
      value: "95%",
      label: "Classification Accuracy",
      description: "PanOptic's performance on subtype identification",
    },
  ];

  return (
    <section id="statistics" className="py-32" ref={ref}>
      <div className="container mx-auto px-6">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <span className="text-primary font-medium text-sm uppercase tracking-widest mb-4 block">
            The Challenge
          </span>
          <h2 className="text-4xl md:text-5xl font-serif font-medium leading-tight max-w-2xl mx-auto">
            Understanding the <span className="italic">urgency</span> of early
            detection.
          </h2>
        </motion.div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 40 }}
              animate={isInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: index * 0.1 }}
              className="relative group"
            >
              <div className="bg-card rounded-2xl p-8 h-full border border-border/50 hover:border-primary/30 transition-colors duration-300">
                <div className="mb-4">
                  <span className="text-4xl md:text-5xl font-serif font-medium text-primary">
                    {stat.value}
                  </span>
                </div>
                <h3 className="font-medium text-lg mb-2">{stat.label}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {stat.description}
                </p>
                
                {/* Decorative element */}
                <div className="absolute top-4 right-4 w-8 h-8 rounded-full bg-secondary opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
