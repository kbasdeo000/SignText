import React from 'react'
import Webcam from "react-webcam";
import FadeIn from 'react-fade-in';
import axios from 'axios';


const ms = require('pretty-ms')
class Translate extends React.Component {
  constructor(props){
    super(props)
    this.state = {
      time: 0,
      isOn: false,
      start: 0,
      screenshot: null,
      getdata: "",
      tab: 0,
      concat: ""
    }
    this.startTimer = this.startTimer.bind(this)
    this.stopTimer = this.stopTimer.bind(this)
    this.resetTimer = this.resetTimer.bind(this)
  }
  
  startTimer() {
    this.setState({
      isOn: true,
      time: this.state.time,
      start: Date.now() - this.state.time,
      screenshot: this.state.screenshot,
      getdata: this.state.getdata,
      concat: this.state.concat
    })
    this.timer = setInterval(() => this.setState({
      time: Date.now() - this.state.start}), 1000);
    this.screenshot = setInterval(() => this.setState({
        screenshot: this.webcam.getScreenshot()} ), 3000);
    this.getdata = setInterval(() => this.setState({
        getdata: this.getdatafunc(), concat: (this.state.concat + this.state.getdata)} ), 3000);
  }

  stopTimer() {
    this.setState({isOn: false})
    this.setState({screenshot: null})
    this.setState({getdata: ""})
    clearInterval(this.timer)
    clearInterval(this.screenshot)
    clearInterval(this.getdata)
  }

  resetTimer() {
    this.setState({time: 0, isOn: false, screenshot: this.state.screenshot, getdata: this.state.getdata, concat: ""})
  }

  getdatafunc() {
    let newcomp = this;
    var bodyFormData = new FormData();
    bodyFormData.append('image', this.webcam.getScreenshot());
    axios({
        method: 'post',
        url: 'http://127.0.0.1:5000/translate',
        data: bodyFormData,
        headers: {'Content-Type': 'multipart/form-data' }
        })
        .then(function (response) {
            console.log(response);
            newcomp.setState({ getdata: response.data});
        })
        .catch(function (response) {
            //handle error
            console.log(response);
        });
  }

  
  render() {
    let start = (this.state.time === 0) ?
    <button onClick={this.startTimer} type="button" class="btn btn-outline-warning btn-lg">Start</button>:
      null
    let stop = (this.state.time === 0 || !this.state.isOn) ?
      null :
      <button onClick={this.stopTimer} type="button" class="btn btn-outline-warning btn-lg">Stop</button>
    let resume = (this.state.time === 0 || this.state.isOn) ?
      null :
      <button onClick={this.startTimer} type="button" class="btn btn-outline-warning btn-lg">Resume</button>
    let reset = (this.state.time === 0 || this.state.isOn) ?
      null :
      <button onClick={this.resetTimer} type="button" class="btn btn-outline-warning btn-lg">Reset</button>
    
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
        <textarea id="TextBox" rows="15" cols="55" value={this.state.concat}>
                   Your translation will appear here.
        </textarea>
        <div>
          <div className='screenshots'>
            {this.state.screenshot ? <img src={this.state.screenshot} height={265} width={355}/> : null}
          </div>
        </div>
      </div>
      <div>
        <h3>Timer: {ms(this.state.time)}</h3>
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


export { Translate};
