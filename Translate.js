import React from 'react'
import SearchField from "react-search-field";
import Webcam from "react-webcam";
import FadeIn from 'react-fade-in';
import axios from 'axios';


// function webcam () {

//   console.log('webcam');
// }


// export const Translate = () => (
//     <FadeIn>
//       <p>
//         .
//       </p>
//     <div>
//       <center>
//         <Webcam audio={false} height={365} width={500} />

//         <textarea id="TextBox" rows="15" cols="55">
//         Your translation will appear here.
//         </textarea>
//       </center>

//       <center>
//      <button type="button" class="btn btn-outline-warning btn-lg"> Start </button>
//      <button type="button" class="btn btn-outline-warning btn-lg"> Stop </button>
//      </center>
//     </div>
//     </FadeIn>
// )

const ms = require('pretty-ms')
class Translate extends React.Component {
  constructor(props){
    super(props)
    this.state = {
      time: 0,
      isOn: false,
      start: 0,
      screenshot: null,
      tab: 0
    }
    this.startTimer = this.startTimer.bind(this)
    this.stopTimer = this.stopTimer.bind(this)
    this.resetTimer = this.resetTimer.bind(this)
  }

  handleClick = () => {
    const screenshot = this.webcam.getScreenshot();
    this.setState({ screenshot });

    var bodyFormData = new FormData();
    bodyFormData.append('image', screenshot);
    axios({
        method: 'post',
        url: 'http://127.0.0.1:5000/translate',
        data: bodyFormData,
        headers: {'Content-Type': 'multipart/form-data' }
        })
        .then(function (response) {
            console.log(response);
            const prediction = response
        })
        .catch(function (response) {
            //handle error
            console.log(response);
        });
  }

  startTimer() {
    this.setState({
      isOn: true,
      time: this.state.time,
      start: Date.now() - this.state.time
    })
    this.timer = setInterval(() => this.setState({
      time: Date.now() - this.state.start
    }), 1);
    const screenshot = this.webcam.getScreenshot();
    this.setState({ screenshot });
  }
  stopTimer() {
    this.setState({isOn: false})
    clearInterval(this.timer)
  }
  resetTimer() {
    this.setState({time: 0, isOn: false})
  }



  render() {
    let start = (this.state.time == 0) ?
    <button onClick={this.startTimer} type="button" class="btn btn-outline-warning btn-lg">Start</button>:
      null
    let stop = (this.state.time == 0 || !this.state.isOn) ?
      null :
      <button onClick={this.stopTimer}>stop</button>
    let resume = (this.state.time == 0 || this.state.isOn) ?
      null :
      <button onClick={this.startTimer}>resume</button>
    let reset = (this.state.time == 0 || this.state.isOn) ?
      null :
      <button onClick={this.resetTimer}>reset</button>
    return(
      <FadeIn>
      <center>
      <div>
        <Webcam
          audio={false}
          height={365}
          width={500}
          ref={node => this.webcam = node}
        />
        <textarea id="TextBox" rows="15" cols="15">
                   Your translation will appear here.
        </textarea>
        <div>
          <div className='screenshots'>
            <div className='controls'>
              <button onClick={this.handleClick} type="button" class="btn btn-outline-warning btn-lg">Start</button>
              <button type="button" class="btn btn-outline-warning btn-lg"> Stop </button>
            </div>
            {this.state.screenshot ? <img src={this.state.screenshot} height={265} width={355}/> : null}
          </div>
        </div>
      </div>
      <div>
        <h3>timer: {ms(this.state.time)}</h3>
        {start}
        {resume}
        {stop}
        {reset}
      </div>
      </center>
      </FadeIn>
    )
  }
}

// class Translate extends React.Component {
//   constructor(props) {
//     super(props);
//     this.state = {
//       screenshot: null,
//       tab: 0
//     };
//   }

//   handleClick = () => {
//     const screenshot = this.webcam.getScreenshot();
//     this.setState({ screenshot });
//   }

//   render() {
//     return (
//       <FadeIn>
//       <center>
//       <div>
//         <Webcam
//           audio={false}
//           height={365}
//           width={500}
//           ref={node => this.webcam = node}
//         />
//         <textarea id="TextBox" rows="15" cols="55">
//                    Your translation will appear here.
//         </textarea>
//         <div>
//           <div className='screenshots'>
//             <div className='controls'>
//               <button onClick={this.handleClick} type="button" class="btn btn-outline-warning btn-lg">Start</button>
//               <button type="button" class="btn btn-outline-warning btn-lg"> Stop </button>
//             </div>
//             {this.state.screenshot ? <img src={this.state.screenshot} height={265} width={355}/> : null}
//           </div>
//         </div>
//       </div>
//       </center>
//       </FadeIn>
//     );
//   }
// }

export { Translate};
