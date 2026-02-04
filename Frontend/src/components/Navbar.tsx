import { motion } from "framer-motion";
import { useState, useEffect } from "react";

const navItems = [
  { label: "About", href: "#about" },
  { label: "Statistics", href: "#statistics" },
  { label: "Research", href: "#research" },
  { label: "Prototype", href: "#prototype" },
];

export const Navbar = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? "bg-glass shadow-soft" : "bg-transparent"
      }`}
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <a href="#" className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span className="font-serif text-xl font-medium text-foreground">
              pan—optic
            </span>
          </a>

          <div className="hidden md:flex items-center gap-8">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="text-sm font-medium text-muted-foreground hover:text-foreground transition-colors tracking-wide uppercase"
              >
                {item.label}
              </a>
            ))}
          </div>

          <a
            href="#prototype"
            className="hidden md:flex items-center gap-2 px-5 py-2.5 rounded-full border border-foreground/20 hover:bg-foreground hover:text-background transition-all duration-300 text-sm font-medium tracking-wide"
          >
            Try Prototype
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
          </a>
        </div>
      </div>
    </motion.nav>
  );
};
