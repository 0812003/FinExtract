import React from 'react';
import { Download, Loader, Upload } from 'lucide-react';

const InfoSection = () => {
  return (
    <div className="min-h-3/4 flex flex-col bg-white">
      <section className="container mx-auto px-4 py-16">
        <h2 className="text-3xl font-bold text-center mb-8 text-[#2c1754]">
          How It Works ?
        </h2>
        <div className="flex flex-col md:flex-row justify-between gap-8">
          <div className="glass-box md:w-1/3">
            <div className="flex items-center"> {/* Flexbox for icon and text alignment */}
              <h3 className="text-2xl font-semibold mb-4 text-[#2c1754] mr-2">Step 1:</h3> {/* Added margin right */}
              <Upload className="h-6 w-6 text-[#2c1754]" /> {/* Adjust size as needed */}
            </div>
            <h4 className='text-xl font-serif mb-2 text-black'>Upload Your File</h4>
            <p className="text-[#2c1754]">Upload your PDF or image file using our secure upload box.</p>
          </div>
          <div className="glass-box md:w-1/3">
            <div className="flex items-center"> {/* Flexbox for icon and text alignment */}
              <h3 className="text-2xl font-semibold mb-4 text-[#2c1754] mr-2">Step 2:</h3> {/* Added margin right */}
              <Loader className="h-6 w-6 text-[#2c1754]" /> {/* Adjust size as needed */}
            </div>
            <h4 className='text-xl font-serif mb-2 text-black'>Process Your File</h4>
            <p className="text-[#2c1754]">Our system will automatically process your file and extract the tables.</p>
          </div>
          <div className="glass-box md:w-1/3">
            <div className="flex items-center"> {/* Flexbox for icon and text alignment */}
              <h3 className="text-2xl font-semibold mb-4 text-[#2c1754] mr-2">Step 3:</h3> {/* Added margin right */}
              <Download className="h-6 w-6 text-[#2c1754]" /> {/* Adjust size as needed */}
            </div>
            <h4 className='text-xl font-serif mb-2 text-black'>Download Your File</h4>
            <p className="text-[#2c1754]">Download the extracted tables in your preferred format.</p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default InfoSection;