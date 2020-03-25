import React from 'react'
import SearchField from "react-search-field";
import Camera from 'react-html5-camera-photo';
import 'react-html5-camera-photo/build/css/index.css';


export const Translate = () => (
    <div>
        <p align="center">
        <SearchField
        placeholder="Search..."
        searchText="Search..."
        classNames="test-class"
        />
        </p>

        <Camera align="left"
         //onTakePhoto = { (dataUri) => { handleTakePhoto(dataUri); } }
    />
        <p align="center">   
        <textarea id="TextBox" rows="15" cols="60">
        Your translation will appear here.
        </textarea>
        </p> 
  );
    </div>
)

function App (props) {
    function handleTakePhoto (dataUri) {
      // Do stuff with the photo...
      console.log('takePhoto');
    }
}
