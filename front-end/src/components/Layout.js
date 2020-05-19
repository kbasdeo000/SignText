import React from 'react';
import Container from 'react-bootstrap/Container';
import styled from 'styled-components';
import backgroundphoto from '../assets/backgroundphoto.jpg';


const Styles = styled.div`
    .overlay {
        background: url(${backgroundphoto}) no-repeat fixed bottom;
        background-color: #888;
        opacity: 1.0;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        z-index: -1;
    }
    .jumbo {
        background-size: cover;
        color: #efefef;
        height: 200px;
        position: relative;
        z-index: -2;
    }
`;

export const Layout = (props) => (
    <Styles> 
        <div className="overlay"></div>
        {/* <div className="jumbo"></div> */}
        <Container>
        {props.children}
        </Container>
    </Styles>

)