// Sorting runs outside the UI thread, including the million-Gaussian setting.
let positions,depth;
self.onmessage=({data})=>{
  if(data.type==='init'){
    positions=new Float32Array(data.positions);depth=new Float32Array(positions.length/3);
    return;
  }
  if(data.type!=='sort'||!positions)return;
  const m=data.matrix,count=depth.length,order=Uint32Array.from({length:count},(_,i)=>i);
  for(let i=0;i<count;i++)depth[i]=m[2]*positions[i*3]+m[6]*positions[i*3+1]+m[10]*positions[i*3+2]+m[14];
  order.sort((a,b)=>depth[a]-depth[b]);
  self.postMessage({type:'sorted',order},[order.buffer]);
};
