import React from 'react'
import Webcam from "react-webcam";
import FadeIn from 'react-fade-in';
import axios from 'axios';


const ms = require('pretty-ms')
class Translate extends React.Component {
  constructor(props){
    super(props)
    this.state = {
      time: 3000,
      isOn: false,
      start: 3000,
      screenshot: null,
      getdata: "",
      tab: 0,
      concat: ""
    }
    this.startTimer = this.startTimer.bind(this)
    this.resetTimer = this.resetTimer.bind(this)
    this.stopTimer = this.stopTimer.bind(this)
  }
  
  startTimer() {
    this.setState({
      isOn: true,
      time: this.state.time,
      start: this.state.start,
      screenshot: this.state.screenshot,
      getdata: this.state.getdata,
      concat: this.state.concat
    })
    this.timer = setInterval(() => this.setState({
      time: this.state.time - 1000}), 1000);
    this.screenshot = setInterval(() => this.setState({
        screenshot: this.webcam.getScreenshot(), time: 3000} ), 3000);
    this.getdata = setInterval(() => this.setState({
        getdata: this.getdatafunc(), concat: (this.state.concat + this.state.getdata)} ), 3000);
  }

  stopTimer() {
    this.setState({ 
      isOn: false,
      time: 3000,
      start: this.state.start,
      screenshot: this.state.screenshot,
      getdata: this.state.getdata,
      concat: this.state.concat})
      clearInterval(this.timer)
      clearInterval(this.screenshot)
      clearInterval(this.getdata)
  }

  resetTimer() {
    this.setState({ 
      isOn: false,
      time: 3000,
      start: this.state.start,
      screenshot: null,
      getdata: "",
      concat: ""})
      clearInterval(this.timer)
      clearInterval(this.screenshot)
      clearInterval(this.getdata)
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
    let start = (!this.state.isOn) ?
    <button onClick={this.startTimer} type="button" class="btn btn-outline-warning btn-lg">Start</button>:
      null 
    let stop = (!this.state.isOn) ?
      null :
      <button onClick={this.stopTimer} type="button" class="btn btn-outline-warning btn-lg">Stop</button>
    let reset = (this.state.isOn) ?
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
        <div>
        <h3>Timer: {ms(this.state.time)}</h3>
        {start}
        {stop}
        {reset}
      </div>
          <div className='screenshots'>
            {this.state.screenshot ? <img src={this.state.screenshot} height={265} width={355}/> : null}
          </div>
        </div>
      </div>
      </center>
      </FadeIn>
    )
  }
}


export { Translate};
