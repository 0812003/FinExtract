import express from "express";
import multer from "multer";
import cors from "cors";
import axios from "axios";
import fs from "fs";
import path from "path";
import FormData from "form-data";
import mongoose from "mongoose";
import dotenv from "dotenv";
import cloudinary from "./lib/cloudinary.js";
import File from "./models/file.model.js"; // Import File model

dotenv.config();
const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

const mongoURI = process.env.MONGODB_URI;
// if (!mongoURI) {
//     console.error("❌ MONGO_URI is not defined in .env file");
//     process.exit(1); // Exit the application if URI is missing
// }

mongoose.connect(mongoURI)
    .then(() => console.log("✅ MongoDB Connected"))
    .catch((err) => console.error("❌ MongoDB Connection Error:", err));

const upload = multer({ dest: "uploads/" });

app.post("/upload", upload.single("file"), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: "No file uploaded" });
    }

    let csvPath;

    try {
        const formData = new FormData();
        formData.append("file", fs.createReadStream(req.file.path), req.file.originalname);

        const response = await axios.post("http://localhost:5001/upload", formData, {
            headers: { ...formData.getHeaders() },
            responseType: "json",
        });
        
        const cloudinaryUrl = response.data.file_url;
        console.log(cloudinaryUrl)
       
        res.json({ fileUrl: cloudinaryUrl });

    } catch (error) {
        console.error("Error processing file:", error);
        res.status(500).json({ error: "Error processing file" });
    } finally {
        if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        if (csvPath && fs.existsSync(csvPath)) fs.unlinkSync(csvPath);
    }
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
