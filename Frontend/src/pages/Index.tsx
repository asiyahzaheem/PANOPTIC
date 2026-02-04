import { Navbar } from "@/components/Navbar";
import { Hero } from "@/components/Hero";
import { About } from "@/components/About";
import { Statistics } from "@/components/Statistics";
import { Research } from "@/components/Research";
import { Prototype } from "@/components/Prototype";
import { Footer } from "@/components/Footer";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <Hero />
      <About />
      <Statistics />
      <Research />
      <Prototype />
      <Footer />
    </div>
  );
};

export default Index;
