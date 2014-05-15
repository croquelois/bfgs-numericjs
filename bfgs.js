"use strict";
(function(exports){
var dot = numeric.dot
var trsp = numeric.transpose;
var mul = numeric.mul;
var sum = numeric.sum;
var add = numeric.add;
var sub = numeric.sub;
var neg = numeric.neg;
var solve = numeric.solve;
var addeq = numeric.addeq;
var subeq = numeric.subeq;
var norm2 = numeric.norm2;
var abs = Math.abs;
function bfgs(guess,obj,maxIter,eps,opt){
  var opt = opt || {};
  var df = obj.df;
  var f = obj.f;
  var B = numeric.identity(guess.length);
  var x = guess.slice();
  var v = {x:x,f:f(x),df:df(x)};

  function lineSearch(v,d){
    var max = 10;
    var p0 = v.f;
    var dp0 = sum(mul(v.df,d));
    var n = opt.maxTry || 25;
    var c1 = opt.c1||0.25;
    var c2 = opt.c2||0.75;
    function zoom(low,high){
      if(low.p === undefined) low.p = f(low.x);
      if(high.p === undefined) high.p = f(high.x);
      while(true){
        var nv = {};
        var a = nv.a = 0.5*(low.a+high.a);
        nv.s = mul(nv.a,d);
        nv.x = add(v.x,nv.s);
        var p = nv.f = f(nv.x);
        if(!n--) throw "too much step, during line search";
        if(p > p0+c1*a*dp0 || p >= low.p){
          high = nv;
        }else{
          nv.df = df(nv.x);
          var dp = nv.dp = sum(mul(nv.df,d));
          if(abs(dp) <= -c2*dp0) return nv;
          if(dp*(high.a-low.a) >= 0) high = low;
          low = nv;
        }
      }
    }
    var nv = {a:1};
    var ov = {x:v.x,a:0};
    var op;
    while(true){
      nv.s = mul(nv.a,d);
      nv.x = add(v.x,nv.s);
      var a = nv.a;
      var p = nv.f = f(nv.x);
      if(p > p0+c1*a*dp0 || (op !== undefined && p > op)) return zoom(ov,nv);
      nv.df = df(nv.x);
      var dp = nv.dp = sum(mul(nv.df,d));
      if(abs(dp) <= -c2*dp0) return nv;
      if(dp >= 0) return zoom(nv,ov);
      ov = nv;
      op = p;
      nv.a = 0.5*(a+max);
      if(!n--) throw "too much step, during line search";
    }
  }

  for(var i=0;i<maxIter;i++){
    var d = neg(solve(B,v.df)); // compute direction
    var og = v.df;
    v = lineSearch(v,d); // compute step length
    /* start: refresh Jacobian */
    var y = sub(v.df,og);
    var vs = trsp([v.s]);
    var ts = [v.s];
    var vy = trsp([y]);
    var ty = [y];
    subeq(B,mul(1/dot(ts,dot(B,vs)),dot(B,dot(vs,dot(ts,B)))));
    addeq(B,mul(1/dot(ty,vs),dot(vy,ty)));
    //console.log(i,v.x,v.df,f(x));
    if(norm2(v.df) < eps) return v;
    /* end: refresh Jacobian */
  }
  return v;
}
exports.bfgs = bfgs;
}(numeric));

