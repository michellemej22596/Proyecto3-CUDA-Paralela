echo "==================================================="
echo "  MEDICIONES DE TRANSFORMADA DE HOUGH - 10 VECES"
echo "==================================================="
echo ""

# Versión 1: Solo Global
echo "=== VERSIÓN 1: MEMORIA GLOBAL ==="
for i in {1..10}; do 
  echo -n "Medición $i: "
  ./hough_basic 2>/dev/null | grep "Tiempo GPU"
done
echo ""

# Versión 2: Global + Constante
echo "=== VERSIÓN 2: GLOBAL + CONSTANTE ==="
for i in {1..10}; do 
  echo -n "Medición $i: "
  ./hough_constante 2>/dev/null | grep "Tiempo GPU"
done
echo ""

# Versión 3: Global + Constante + Compartida
echo "=== VERSIÓN 3: GLOBAL + CONSTANTE + COMPARTIDA ==="
for i in {1..10}; do 
  echo -n "Medición $i: "
  ./hough_compartida 2>/dev/null | grep "Tiempo GPU"
done
echo ""
echo "==================================================="
echo "  MEDICIONES COMPLETADAS"
echo "==================================================="
