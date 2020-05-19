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

export const Home = () => (
    <FadeIn>
    <div className="jumbo"> </div>
    <div>
        <center>
            <h2 style={{color: "#fcba03", fontSize: 50, fontFamily: "Georgia"}}> SignText </h2>
            <p> </p>
            <h1 style={{color: "white", fontSize: 80, fontFamily: "Georgia"}}>A New Kind of Translation.</h1>
            <a href="/translate"><button type="button" class="btn btn-outline-warning btn-lg">Get Started</button></a>
        </center>
        
    </div>
    </FadeIn>
)