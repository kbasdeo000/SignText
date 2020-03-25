import React from 'react';
import { Jumbotron as Jumbo, Container } from 'react-bootstrap';
import styled from 'styled-components';
import aslImage from '../assets/aslphoto.jpg';
import backgroundphoto from '../assets/backgroundphoto.jpg';

const Styles = styled.div`
    .jumbo {
        background: url(${aslImage}) no-repeat fixed bottom;
        background-size: cover;
        color: #efefef;
        height: 200px;
        position: relative;
        z-index: -2;
    }

    .overlay {
        background: url(${backgroundphoto}) no-repeat fixed bottom;
        background-color: #888;
        opacity: 0.4;
        position: absolute;
        top: 0;
        left: 0;
        bottom: 0;
        right: 0;
        z-index: -1;
    }
`;

export const Jumbotron = () => (
    <Styles> 
        <Jumbo fluid className="jumbo">
            <div className="overlay"></div>
            <Container>

            </Container>
        </Jumbo>
    </Styles>

)