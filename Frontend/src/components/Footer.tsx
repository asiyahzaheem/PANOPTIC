import { motion } from "framer-motion";

export const Footer = () => {
  return (
    <footer className="py-16 border-t border-border">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row items-center justify-between gap-8">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span className="font-serif text-xl font-medium">pan—optic</span>
          </div>

          <div className="flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#about" className="hover:text-foreground transition-colors">
              About
            </a>
            <a href="#statistics" className="hover:text-foreground transition-colors">
              Statistics
            </a>
            <a href="#research" className="hover:text-foreground transition-colors">
              Research
            </a>
            <a href="#prototype" className="hover:text-foreground transition-colors">
              Prototype
            </a>
          </div>

          <p className="text-sm text-muted-foreground">
            © 2024 PanOptic. Research prototype.
          </p>
        </div>
      </div>
    </footer>
  );
};
