function makeTree(d) {
  if (d > 0) {
    d -= 1;
    return [makeTree(d), makeTree(d)];
  }
  return [null, null];
}

function checkTree(node) {
  const [l, r] = node;
  if (l === null) {
    return 1;
  } else {
    return 1 + checkTree(l) + checkTree(r);
  }
}

function makeCheck(d, make = makeTree, check = checkTree) {
  return check(make(d));
}

function main(n, minDepth = 4) {
  const maxDepth = Math.max(minDepth + 2, n);
  const stretchDepth = maxDepth + 1;

  console.log(
    `stretch tree of depth ${stretchDepth}\t check: ${makeCheck(stretchDepth)}`
  );

  const longLivedTree = makeTree(maxDepth);
  const mmd = maxDepth + minDepth;

  for (let d = minDepth; d < stretchDepth; d++) {
    const i = 2 ** (mmd - d);
    let cs = 0;

    for (let j = 0; j < i; j++) {
      cs += makeCheck(d);
    }

    console.log(`${i}\t trees of depth ${d}\t check: ${cs}`);
  }

  console.log(
    `long lived tree of depth ${maxDepth}\t check: ${checkTree(longLivedTree)}`
  );
}

// Get command line argument or use default value
const n = process.argv.length > 2 ? parseInt(process.argv[2]) : 16;
main(n);