import mongoose from "mongoose";
const MONGOLAB_URI = "mongodb+srv://user:user@cluster0.a87ri9o.mongodb.net/?retryWrites=true&w=majority";

// Connecting database
const connectDB = async () => {
  try {
    await mongoose.connect(MONGOLAB_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log(`Database Connected with ${mongoose.connection.host}`);
  } catch (error) {
    console.error(error);
  }
};

export default connectDB; // Export the connectDB function
