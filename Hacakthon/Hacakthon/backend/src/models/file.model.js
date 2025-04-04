import mongoose from "mongoose";

const FileSchema = new mongoose.Schema({
    originalName: String,
    cloudinaryUrl: String,
    uploadedAt: { type: Date, default: Date.now }
});

export default mongoose.model("File", FileSchema);
