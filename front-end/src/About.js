import React from 'react'
import FadeIn from 'react-fade-in';
// import styled from 'styled-components';

// const Styles = styled.div`

//     .jumbo {
//         background-size: cover;
//         color: #efefef;
//         height: 200px;
//         position: relative;
//         z-index: -2;
//     }
// `;

export const About = () => (
    <FadeIn>
            <div className="jumbo"> </div>
    <div>
        <center>
            <h2 style={{color: "#fcba03", fontSize: 30, fontFamily: "Georgia"}}> About SignText </h2>
            <p style={{color: "white", fontSize: 20, fontFamily: "Georgia"}}><i>Our Vision</i></p>
            <p style={{color: "white", fontSize: 20, fontFamily: "Georgia"}}>  SignText translates American Sign Language (ASL) into the written English alphabet, built from a machine learning model of our creation. Our goals are to give the DDH community another means of communication and to bridge the language gap between hearing and non-hearing peoples. </p>
        </center>
    </div>
    </FadeIn>
)