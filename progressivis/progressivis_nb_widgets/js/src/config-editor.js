"use strict";
import React, { Component } from "react";
import ReactDOM from "react-dom";
import { elementReady } from "./es6-element-ready";
import $ from 'jquery';

function getOnly(dict, keys){
    var res = {};
      keys.forEach((e)=>{res[e] = dict[e]})
      return res
      }
    function renderSelect(val, opts, chg){
        let k;
        return (<select  value={val} onChange={chg}>
            {Object.keys(opts).map((k, _)=>{return <option key={k} value={k}>{opts[k]}</option>})}
            </select>)
    }
    function renderHeader(title, label, dict, val, handler){
        return (<tbody  key={title}><tr><td colSpan="2"><h3>{title}</h3></td></tr><tr><td><label>{label}&nbsp;</label></td><td>{renderSelect(val, dict, handler)}</td></tr></tbody>)
    }
    function renderNamedSelect(label, dict, val, handler){
        return (<tbody key={label}><tr><td><label>{label}&nbsp;</label></td><td>{renderSelect(val, dict, handler)}</td></tr></tbody>)
    }

    function renderInput(label, type, name, val, handler){
        return (<tbody key={name}><tr><td><label>{label}&nbsp;</label></td><td><input  value={val} name={name} type={type} onChange={handler}/></td></tr></tbody>)
    }

    class Rebin extends React.Component {
        constructor(props) {
            super(props);
            this.handleChange = this.handleChange.bind(this);
        }
        handleChange(key) {
            return (val) => {this.props.handleChange(key, val);};
        }
        render(){
            let typeDict = {none: "None", square: "Square", rect: "Rect", voronoi: "Voronoi"};
            let rebinType = this.props.value['type'];
            let res = [renderHeader("Rebin", "Type", typeDict, rebinType , this.handleChange('type'))];
            if (rebinType=="none") return  (<>{res}</>);
            let aggrDict = {max: 'Max', mean:'Mean', sum: 'Sum', min: 'Min'};
            res.push(renderNamedSelect("Aggregate", aggrDict, this.props.value['aggregation']  , this.handleChange('aggregation')));
            //renderInput(label, type, name, val, handler)
            if(rebinType=="square") {
                res.push(renderInput("Size", "number", "rebinSize",this.props.value['size'] , this.handleChange('size')))
            };
            if(rebinType=="rect") {
                res.push(renderInput("Width", "number", "rebinWidth",this.props.value['width'] , this.handleChange('width')))
                res.push(renderInput("Height", "number", "rebinHeight",this.props.value['height'] , this.handleChange('height')))
            };
            if(rebinType=="voronoi") {
                res.push(renderInput("Size", "number", "rebinSize",this.props.value['size'] , this.handleChange('size')))
                res.push(renderInput("Stroke", "text", "rebinStroke",this.props.value['stroke'] , this.handleChange('stroke')))
            };
            return  (<>{res}</>);
        }
    }
    class Rescale extends React.Component {
        constructor(props) {
            super(props);
            this.handleChange = this.handleChange.bind(this);
        }
        handleChange(key) {
            return (val) => {this.props.handleChange(key, val);};
        }
        render(){
            let typeDict = {linear: "Linear",
                log: "Log", sqrt: "Square Root",
                cbrt:"Cubic Root", equidepth: "Equi-depth"};
            let rescaleType = this.props.value['type'];
            let res = [renderHeader("Rescale", "Type", typeDict, rescaleType , this.handleChange('type'))];
            if(rescaleType == 'equidepth'){
                res.push(renderInput("Size", "number", "rescaleLevels",this.props.value['levels'] , this.handleChange('levels')))
            }
            return  (<>{res}</>);
       }
     }

    class Compose extends React.Component {
        constructor(props) {
            super(props);
            this.handleChange = this.handleChange.bind(this);
        }
        handleChange(key) {
            return (val) => {this.props.handleChange(key, val);};
        }
        render(){
            let compDict = {none: "none", invmin: "Invmin", mean: "Mean", max: "Max",
                    blend: "Blend", weaving: "Weaving", /*propline: "Propline",
                    hatching: "Hatching",*/ separate: "Separate", glyph: "Glyph",
                    dotdensity: "Dotdensity", time: "Time"};
            let compValue = this.props.value['mix'];
            let res = [renderHeader("Compose", "Mix", compDict, compValue, this.handleChange('mix'))];
            if(compValue in ["none", "mean", "max", "separate"]) return  (<>{res}</>);
            if(compValue == "invmin"){
                res.push(renderInput("Threshold", "number", "compThreshold",this.props.value['threshold'] , this.handleChange('threshold')))
            }
            if(compValue == "blend"){
                let mixingDict = {additive: "Additive", multiplicative: "Multiplicative"};
                res.push(renderNamedSelect("Mixing", mixingDict, this.props.value['mixing']  , this.handleChange('mixing')));
            }
            if(compValue == "weaving"){
                let weavingDict = {square: "Square", random: "Random", hexagon: "Hexagon", triangle: "Triangle"};
                res.push(renderNamedSelect("Weaving", weavingDict, this.props.value['weaving']  , this.handleChange('weaving')));
                res.push(renderInput("Size", "number", "compSize",this.props.value['size'] , this.handleChange('size')));
            }
            if(compValue == "glyph"){
                let templDict = {punchcard: "punchcard", bars: "bars"};
                res.push(renderNamedSelect("Template", templDict, this.props.value['template']  , this.handleChange('glyph')));
                res.push(renderInput("Width", "number", "compWidth",this.props.value['width'] , this.handleChange('width')));
                res.push(renderInput("Height", "number", "compHeight",this.props.value['height'] , this.handleChange('height')));
            }
            if(compValue == "dotdensity"){
                res.push(renderInput("Size", "number", "compSize",this.props.value['size'] , this.handleChange('size')))
            }
            if(compValue == "time"){
                res.push(renderInput("Interval(s)", "number", "compInterval",this.props.value['interval'] , this.handleChange('interval')))
            }
            return  (<>{res}</>);





       }
    }
      
   class ConfigForm extends React.Component {
      constructor(props){
          super(props);
          this.handleRebin = this.handleRebin.bind(this);
          this.handleRescale = this.handleRescale.bind(this);
          this.handleCompose = this.handleCompose.bind(this);
          this.handleGrp = this.handleGrp.bind(this);
          this.tidy = this.tidy.bind(this);
          this.state = {rebin: {type: "none", aggregation: "max", size: 4, width: 4, height: 4, stroke: 'rgba(0, 0, 0, .1)'}, rescale: {type: "cbrt", levels: 4}, compose: {mix: "max", threshold: 1, size: 8, width: 32, height: 32, mixing:"additive", shape: "square", template: "punchcard", interval: 0.6}, data: {}, legend: true};
      }
      handleGrp(grp, key, evt){
          let stateCopy = Object.assign({}, this.state);
          stateCopy[grp][key] = evt.target.value;
          this.setState(stateCopy);
          window.spec = this.tidy(Object.assign({}, stateCopy));
      };
      handleRebin(key, val){return this.handleGrp('rebin', key, val);};
      handleRescale(key, val){return this.handleGrp('rescale', key, val);};
      handleCompose(key, val){return this.handleGrp('compose', key, val);};
      renderRebin(){
          return (<Rebin value={this.state.rebin} handleChange={this.handleRebin}/>)
      }
      renderRescale(){
          return (<Rescale value={this.state.rescale} handleChange={this.handleRescale}/>)
      }
      renderCompose(){
          return (<Compose value={this.state.compose} handleChange={this.handleCompose}/>)
      }
      tidy(){
          let stateCopy = Object.assign({}, this.state);
          // Rebin
          if(stateCopy.rebin.type == 'none') stateCopy.rebin = {type: 'none'};
          if(stateCopy.rebin.type == 'rect') stateCopy.rebin = getOnly(stateCopy.rebin, ['type', 'aggregation', 'width', 'height']);
          if(stateCopy.rebin.type == 'square') stateCopy.rebin = getOnly(stateCopy.rebin, ['type', 'aggregation', 'size']);
          if(stateCopy.rebin.type == 'voronoi') stateCopy.rebin = getOnly(stateCopy.rebin, ['type', 'aggregation', 'size', 'stroke']);
          // Rescale
          if(stateCopy.rescale.type != 'equidepth') stateCopy.rescale = getOnly(stateCopy.rescale, ['type']);
          // Compose
          if(stateCopy.compose.mix in ["none", "mean", "max", "separate"]) stateCopy.compose = getOnly(stateCopy.compose, ['mix']);
          if(stateCopy.compose.mix == 'invmean') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'threshold']);
          if(stateCopy.compose.mix == 'blend') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'mixing']);
          if(stateCopy.compose.mix == 'weaving') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'size']);
          if(stateCopy.compose.mix == 'glyph') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'template', 'width', 'height']);
          if(stateCopy.compose.mix == 'dotdensity') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'size']);
          if(stateCopy.compose.mix == 'time') stateCopy.compose = getOnly(stateCopy.compose, ['mix', 'interval']);
          return stateCopy;
      }
     render() {
        window.spec = this.tidy();
      return (<form id="editform"><table>
        {this.renderRebin()}
        {this.renderRescale()}
        {this.renderCompose()}
	      </table>
        <div hidden>{JSON.stringify(this.tidy())}</div>
	      </form>)
      }
    }
    elementReady("#root").then((_)=>{
      ReactDOM.render(
      <ConfigForm />, 
      document.getElementById("root")
      );
    });
// adding Configure button
elementReady("#mdm-form").then((_)=>{
    //console.log("Config button", $("#config-btn"));
    $("#config-btn").click(function(){var sty = $("#mdm-form").css("display");
					  var newSty = sty=="none" ? "block":"none";
					  var txt = newSty=="none" ? "Configure...":"Close props editor";
					  $("#mdm-form").css("display", newSty);
					  $("#config-btn").text(txt);

					 });
    });
