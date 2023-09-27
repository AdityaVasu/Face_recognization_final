import mongoose from "mongoose";

const lawyerSchema = new mongoose.Schema({
    BarcouncilNO: {
        type: String,
        required: true,
        unique: true, // Ensures each BarcouncilNO is unique
      },
      name: {
        type: String,
    
      },
    profileImage: {
    type: String,
    default:"https://cdn3.vectorstock.com/i/1000x1000/50/27/lawyer-icon-male-user-person-profile-avatar-vector-20905027.jpg",
  },

});

export const LawyerModel = mongoose.model("Lawyer",lawyerSchema);