import React from 'react';
import { Instagram, Youtube } from 'lucide-react'; // Import the Instagram icon

const Footer = () => {
  return (
    <footer className="bg-[#ffffff] py-4 px-10 text-black"> {/* Dark background, padding, white text */}
      <div className="container mx-auto flex flex-col md:flex-row justify-between items-center"> {/* Responsive layout */}
        <div className="text-sm">
          &copy; {new Date().getFullYear()} FinExtract. All rights reserved. {/* Current year */}
        </div>
        <div className="flex items-center space-x-4"> {/* Social media links */}
            <h2>Follow Us On</h2>
          <a href="your-instagram-link" target="_blank" rel="noopener noreferrer" className="hover:text-gray-300 transition">
            <Instagram className="h-6 w-6" />
          </a>
          <a href="your-instagram-link" target="_blank" rel="noopener noreferrer" className="hover:text-gray-300 transition">
            <Youtube className="h-8 w-6" />
          </a>
          {/* Add more social media icons here if needed */}
        </div>
      </div>
    </footer>
  );
};

export default Footer;