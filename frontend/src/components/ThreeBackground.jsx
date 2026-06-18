import { useEffect, useRef } from "react";
import * as THREE from "three";

// Hafif, dönen wireframe + parçacık arka planı. İçeriğin arkasında durur.
export default function ThreeBackground() {
  const ref = useRef(null);

  useEffect(() => {
    const mount = ref.current;
    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
    camera.position.z = 6;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    // dönen wireframe ikosahedron
    const geo = new THREE.IcosahedronGeometry(2.2, 1);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xc45a26,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
    });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);

    // parçacıklar
    const count = 400;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count * 3; i++) positions[i] = (Math.random() - 0.5) * 22;
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const points = new THREE.Points(
      pGeo,
      new THREE.PointsMaterial({ color: 0x4a5070, size: 0.05 })
    );
    scene.add(points);

    let raf;
    const animate = () => {
      mesh.rotation.x += 0.0015;
      mesh.rotation.y += 0.002;
      points.rotation.y += 0.0006;
      renderer.render(scene, camera);
      raf = requestAnimationFrame(animate);
    };
    animate();

    const onResize = () => {
      const nw = mount.clientWidth;
      const nh = mount.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      geo.dispose();
      mat.dispose();
      pGeo.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
    };
  }, []);

  return <div className="three-bg" ref={ref} />;
}
