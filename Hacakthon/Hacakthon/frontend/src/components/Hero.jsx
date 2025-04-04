import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import Navbar from './Navbar';
import UploadBox from './UploadBox';

const Home = () => {
  const navigate = useNavigate();

  const handleButtonClick = () => {
    navigate('/complaint');
  };

  return (
    <div className="bg-gradient-to-r from-[#641f34] to-[#2c1754] min-h-screen">
      <Navbar />

      <div className="container mx-auto px-4 py-6 flex flex-col items-center">
        <div className="max-w-3xl mx-auto w-full text-center">
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-3 leading-tight"> 
        Extract Financial Ledgers from PDFs/Images
      </h1>
      <p className="text-lg md:text-xl text-gray-200 mb-5 leading-relaxed">
      Save time and ensure accuracy with our free Financial Ledger Extractor tool. Effortlessly extract financial data, transactions, and tables from PDFs and images in real-time with 100% precision. Simplify your accounting and bookkeeping with seamless data extraction
      </p>
          <div className="flex justify-center">
          <UploadBox/>
         </div>
        </div>  
      </div>
    </div>
  );
};

export default Home;