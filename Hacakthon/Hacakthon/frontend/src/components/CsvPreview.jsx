import React, { useState, useEffect } from "react";
import Papa from "papaparse";

const CSVPreview = ({ csvUrl }) => {
    const [data, setData] = useState([]);

    useEffect(() => {
        if (csvUrl) {
            fetch(csvUrl)
                .then((response) => response.text())
                .then((csvText) => {
                    Papa.parse(csvText, {
                        header: true,
                        skipEmptyLines: true,
                        complete: (result) => {
                            setData(result.data);
                        },
                    });
                })
                .catch((error) => console.error("Error fetching CSV:", error));
        }
    }, [csvUrl]);

    return (
        <div className="max-w-5xl mx-auto mt-10 p-8 bg-white shadow-xl rounded-3xl">
            <h2 className="text-4xl font-bold text-[#641f34] mb-6 text-center">
                CSV File Preview
            </h2>
            {data.length > 0 ? (
                <div className="overflow-x-auto rounded-lg shadow-lg border border-gray-200">
                    <table className="w-full border border-gray-300">
                        <thead>
                            <tr className="bg-[#641f34] text-white">
                                {Object.keys(data[0]).map((key) => (
                                    <th
                                        key={key}
                                        className="px-6 py-4 text-lg font-semibold text-left border border-gray-400"
                                    >
                                        {key}
                                    </th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {data.map((row, index) => (
                                <tr
                                    key={index}
                                    className={`${
                                        index % 2 === 0
                                            ? "bg-gray-100"
                                            : "bg-white"
                                    } hover:bg-[#f8d8e0] transition-all`}
                                >
                                    {Object.values(row).map((value, i) => (
                                        <td
                                            key={i}
                                            className="px-6 py-4 text-gray-800 text-md border border-gray-400"
                                        >
                                            {value}
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            ) : (
                <p className="text-center text-gray-600 text-xl">
                    Loading CSV data...
                </p>
            )}
        </div>
    );
};

export default CSVPreview;
