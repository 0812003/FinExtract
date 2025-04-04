import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const UploadBox = () => {
  const [file, setFile] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [downloadURL, setDownloadURL] = useState(null);
  const [previewFileURL, setPreviewFileURL] = useState(null); 
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const MAX_SIZE = 20 * 1024 * 1024; 

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;

    if (selectedFile.size > MAX_SIZE) {
      alert("File size exceeds 20MB.");
      return;
    }

    if (selectedFile.type === "application/pdf" || selectedFile.type.startsWith("image/")) {
      setFile(selectedFile);
      setDownloadURL(null);
      setPreviewFileURL(null); 

      if (selectedFile.type.startsWith("image/")) {
        const url = URL.createObjectURL(selectedFile);
        setPreviewURL(url);
      } else {
        setPreviewURL(null);
      }
    } else {
      alert("Please select a valid PDF or image file.");
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert("No file selected.");
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percentCompleted);
        },
      });

      setUploadProgress(100);

      if (response.data.fileUrl) {
        console.log(response.data.fileUrl)
        setDownloadURL(response.data.fileUrl);
        setPreviewFileURL(response.data.fileUrl); 
      }
    } catch (error) {
      console.error("Upload failed:", error);
      alert("Error processing file");
    } finally {
      setUploading(false);
    }
  };

  const handlePreview = () => {
    navigate(`/preview`,{ state: { fileUrl: previewFileURL } });
  };

  useEffect(() => {
    return () => {
      if (previewURL) URL.revokeObjectURL(previewURL);
    };
  }, [previewURL]);

  return (
    <div className="bg-gradient-to-r from-[#2c1754] to-[#641f34] p-4 rounded-lg w-full md:w-3/4 flex flex-col items-center">
      <div
        className="border-2 border-dashed rounded-lg p-8 flex flex-col items-center cursor-pointer bg-white w-full h-auto md:h-40"
        onClick={() => fileInputRef.current.click()}
      >
        <input
          type="file"
          accept=".pdf, image/*"
          className="hidden"
          ref={fileInputRef}
          onChange={(e) => handleFileSelect(e.target.files[0])}
        />
        
        {!file && (
          <div>
            <p className="text-gray-800 text-center mb-4">Drop your file here, or browse</p>
            <p className="text-gray-500 text-center">Supports PDF and Image only (20MB max)</p>
          </div>
        )}

        {file && (
          <div className="text-center">
            <p className="text-lg font-semibold mb-2">{file.name}</p>
            {file.type.startsWith("image/") && previewURL && (
              <img src={previewURL} alt={file.name} className="max-w-full h-auto mt-4 rounded-lg shadow-md" />
            )}
          </div>
        )}
      </div>

      {file && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          className="bg-blue-500 text-white px-4 py-2 mt-4 rounded-md hover:bg-blue-600 transition"
        >
          {uploading ? "Uploading..." : "Upload File"}
        </button>
      )}

      {uploading && (
        <div className="w-full bg-gray-200 rounded-full mt-3">
          <div
            className="bg-blue-500 text-xs font-medium text-white text-center p-1 leading-none rounded-full transition-all"
            style={{ width: `${uploadProgress}%` }}
          >
            {uploadProgress}%
          </div>
        </div>
      )}

      {downloadURL && (
        <div className="flex space-x-4 mt-4">
          <button
            onClick={handlePreview}
            className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600 transition"
          >
            Preview File
          </button>
          <a
            href={downloadURL}
            download="processed_data.csv"
            className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 transition"
          >
            Download Processed File
          </a>
        </div>
      )}
    </div>
  );
};

export default UploadBox;
