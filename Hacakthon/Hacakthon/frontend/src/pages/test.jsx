import React, { useState, useEffect } from 'react';

const NeuralGrid = () => {
  const [nodes, setNodes] = useState([]);
  const [mousePosition, setMousePosition] = useState({ x: 50, y: 50 });
  
  useEffect(() => {
    // Create nodes in a more uniform grid pattern
    const rows = 6;
    const cols = 8;
    const newNodes = [];
    
    // Create nodes in a grid with random offset
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        // Base position
        const baseX = (col / (cols - 1)) * 100;
        const baseY = (row / (rows - 1)) * 100;
        
        // Add random offset (±5%)
        const x = Math.max(0, Math.min(100, baseX + (Math.random() - 0.5) * 10));
        const y = Math.max(0, Math.min(100, baseY + (Math.random() - 0.5) * 10));
        
        newNodes.push({
          x,
          y,
          size: 2 + Math.random() * 1.5,
          pulseDelay: Math.random() * 4
        });
      }
    }

    // Create connections between nodes
    newNodes.forEach(node => {
      node.connections = [];
      newNodes.forEach(target => {
        if (node !== target) {
          const distance = Math.hypot(node.x - target.x, node.y - target.y);
          if (distance < 25) { // Adjust connection distance for better coverage
            node.connections.push({
              targetX: target.x,
              targetY: target.y,
              animationDelay: Math.random() * 4,
              opacity: Math.max(0.2, 1 - distance / 25)
            });
          }
        }
      });
    });

    setNodes(newNodes);
  }, []);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    setMousePosition({ x, y });
  };

  const getNodeOffset = (nodeX, nodeY) => {
    const deltaX = (mousePosition.x - 50) / 50;
    const deltaY = (mousePosition.y - 50) / 50;
    const distance = Math.hypot(nodeX - mousePosition.x, nodeY - mousePosition.y);
    const influence = Math.max(0, 1 - distance / 40);
    
    return {
      x: nodeX + deltaX * 3 * influence,
      y: nodeY + deltaY * 3 * influence
    };
  };
return(
  <div className="relative h-screen"> {/* Make the main div relative */}
      <Navbar /> {/* Place Navbar at the top */}

      {/* Hero Section */}
      <div className="container mx-auto px-4 py-6 flex flex-col items-center relative z-10"> {/* Add relative z-10 */}
        <div className="max-w-3xl mx-auto w-full text-center">
          <h1 className="text-4xl md:text-5xl font-bold text-white mb-3 leading-tight">
            Extract Financial Ledgers from PDFs/Images
          </h1>
          <p className="text-lg md:text-xl text-gray-200 mb-5 leading-relaxed">
            Save time and ensure accuracy with our free Financial Ledger Extractor tool.
            Effortlessly extract financial data, transactions, and tables from PDFs and
            images in real-time with 100% precision. Simplify your accounting and
            bookkeeping with seamless data extraction
          </p>
          <div className="flex justify-center">
            <UploadBox />
          </div>
        </div>
      </div>

      {/* Neural Network Background - Now positioned absolutely */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-r from-[#641f34] to-[#2c1754]" />
        <div className="absolute inset-0">
          <svg className="w-full h-full">
            {/* ... (Your existing SVG and animation code) */}
          </svg>
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[#2c1754] opacity-30" />
        </div>
      </div>

      <style jsx>{`
        @keyframes pulseNode {
          0%, 100% {
            opacity: 0.8;
            transform: scale(1);
          }
          50% {
            opacity: 1;
            transform: scale(1.2);
          }
        }

        @keyframes pulseGlow {
          0%, 100% {
            opacity: 0.1;
            transform: scale(1);
          }
          50% {
            opacity: 0.2;
            transform: scale(1.5);
          }
        }

        @keyframes pulseConnection {
          0%, 100% {
            opacity: 0.25;
          }
          50% {
            opacity: 0.4;
          }
        }
      `}</style>
    </div>
  );
};

export default NeuralGrid;