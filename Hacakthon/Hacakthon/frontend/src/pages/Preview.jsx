import React, { useState, useEffect } from "react";
import CSVPreview from "../components/CsvPreview";
import { useLocation } from "react-router-dom";

const FileUpload = () => {
    const [csvUrl, setCsvUrl] = useState("");
    const location = useLocation();
    const { fileUrl } = location.state || {};

    useEffect(() => {
        if (fileUrl) {
            setCsvUrl(fileUrl);
        }
    }, [fileUrl]);

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#2c1754] to-[#641f34] flex items-center justify-center p-6">
            <div className="w-full max-w-4xl p-8 bg-white rounded-3xl shadow-2xl ">
                {csvUrl ? (
                    <CSVPreview csvUrl={csvUrl} />
                ) : (
                    <div className="text-center">
                        <p className="text-gray-600 text-2xl font-semibold">
                            Loading preview...
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FileUpload;
