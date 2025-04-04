import React from 'react';
import Navbar from './Navbar';
import UploadBox from './Uploadbox';
import InfoSection from './InfoSection';
import { Star } from 'lucide-react'; // Import the Star icon

const GenericSection = ({ title, children }) => (
    <section className="container mx-auto px-4 py-16">
    <h2 className="text-3xl font-bold text-center mb-8 text-[#ffffff]">
      {title}
    </h2>
    {children}
  </section>

);

const Review = () => {
  return (
    <div className="min-h-1/2 flex flex-col bg-gradient-to-r from-[#641f34] to-[#2c1754]">
      {/* ... (Navbar, HeroSection, InfoSection remain the same) */}

      {/* Review Section */}
      <GenericSection title="What Our Clients Say">
        <div className="flex flex-col md:flex-row justify-between gap-8">
          <div className="glass-box md:w-1/3">
            <div className="flex items-center mb-2"> {/* Star rating */}
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400" />
            </div>
            <p className="text-[#ffffff] italic">"Excellent service! I highly recommend it." - John Doe</p>
          </div>
          <div className="glass-box md:w-1/3">
            <div className="flex items-center mb-2"> {/* Star rating */}
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400" />
            </div>
            <p className="text-[#ffffff] italic">"Fast and accurate.  Just what I needed." - Jane Smith</p>
          </div>
          <div className="glass-box md:w-1/3">
            <div className="flex items-center mb-2"> {/* Star rating */}
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400 mr-1" />
              <Star className="h-5 w-5 text-yellow-400" />
            </div>
            <p className="text-[#ffffff] italic">"Great tool!  Makes my work so much easier." - David Lee</p>
          </div>
        </div>
      </GenericSection>
    </div>
  );
};

export default Review;