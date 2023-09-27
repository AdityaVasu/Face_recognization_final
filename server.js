import connectDB from "./database.js";
import express from "express";
import {LawyerModel} from "./scheme.js"

const port = 5000;
const app = express();
connectDB()

// Middleware for parsing JSON
app.use(express.json());

// Middleware for parsing URL-encoded data
app.use(express.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.send("Hi");
});
app.post("/register", async (req, res) => {
    try {
      const { BarcouncilNO, name, profileImage } = req.body;
  
      // Check if the user with the same BarcouncilNO already exists
      const existingUser = await LawyerModel.findOne({ BarcouncilNO });
  
      if (existingUser) {
        return res.status(400).json({ message: "User with this BarcouncilNO already exists." });
      }
  
      // Create a new user
      const user = new LawyerModel({
        BarcouncilNO,
        name,
        profileImage,
      });
  
      // Save the user to the database
      await user.save();
  
      res.status(201).json({ message: "User registered successfully.", user });
    } catch (error) {
      console.error("Error while registering user:", error);
      res.status(500).json({ message: "Internal server error." });
    }
  });

app.listen(port, () => {
  console.log(`Listening at ${port}`);
});
