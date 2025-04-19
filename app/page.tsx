import React from "react";
import { Card, CardContent } from "../components/ui/card";
import { Button } from "../components/ui/button";
import { Github } from "lucide-react";
import { motion } from "framer-motion";

export default function HPCPortfolio() {
  return (
    <main className="bg-white text-gray-900 p-6 space-y-12">
      <section className="text-center space-y-2">
        <h1 className="text-4xl font-bold">Speeding Up Intelligence</h1>
        <p className="text-lg text-gray-600">An HPC Exploration by Anshika Gaur</p>
        <Button asChild size="lg">
          <a
            href="https://github.com/gauranshika29/hpc-neural-network"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Github className="mr-2" /> View on GitHub
          </a>
        </Button>
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardContent className="p-4 space-y-2">
            <h2 className="text-2xl font-semibold">Project Summary</h2>
            <p>
              This project explores how neural networks can be optimized using parallelism.
              Built in C++ from scratch, the model leverages OpenMP to dramatically reduce
              training time while maintaining accuracy.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 space-y-2">
            <h2 className="text-2xl font-semibold">Speedup Results</h2>
            <ul className="list-disc list-inside">
              <li>Without OpenMP: 402 ms</li>
              <li>With OpenMP: 147 ms</li>
              <li>Speedup: ~2.73×</li>
            </ul>
          </CardContent>
        </Card>
      </section>

      <section className="space-y-4">
        <h2 className="text-2xl font-bold">Screenshots</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <img src="/omp_snippet.png" alt="OpenMP Snippet" className="rounded-xl shadow" />
          <img src="/parallel_output2.png" alt="Parallel Output" className="rounded-xl shadow" />
          <img src="/serial_output2.png" alt="Serial Output" className="rounded-xl shadow" />
          <img src="/github_push.png" alt="GitHub Push" className="rounded-xl shadow" />
        </div>
      </section>

      <motion.section
        className="text-center pt-12"
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <p className="text-lg">Proudly created by Anshika Gaur</p>
      </motion.section>
    </main>
  );
}
