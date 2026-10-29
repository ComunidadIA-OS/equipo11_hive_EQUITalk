# EQUITalk – Plan de puesta en producción, permisos y garantías de uso responsable

## 1. Objetivo y alcance

EQUITalk es una herramienta de código abierto que analiza grabaciones de voz para estudiar la distribución de la participación en conversaciones. Su objetivo es ayudar a detectar posibles desigualdades en las interacciones, por ejemplo, en reuniones de trabajo, grupos universitarios, conversaciones entre amistades o tertulias de radio y televisión.

La aplicación utiliza técnicas de inteligencia artificial para transcribir el audio, identificar los turnos de habla, estimar el género percibido de las voces y calcular métricas como el tiempo de intervención, los solapamientos, las interrupciones, los monólogos prolongados y las diferencias de participación entre grupos.

La solución se plantea principalmente como una aplicación de ejecución local en el equipo del usuario (CLI o interfaz web local mediante Gradio). De cara al futuro, para aquellos dispositivos que no dispongan de recursos de GPU suficientes, se contempla la posibilidad de incorporar opcionalmente una alternativa basada en una API externa de inferencia en la Unión Europea. Esta opción futura requerirá siempre un aviso previo y transparente antes de enviar datos fuera del dispositivo.

El propósito de EQUITalk es facilitar el análisis y la reflexión sobre las dinámicas conversacionales, no identificar a personas concretas, determinar su género real ni emitir conclusiones definitivas sobre sus comportamientos. Las estimaciones y métricas deben interpretarse como indicadores orientativos, sujetos a errores y sesgos.

## 2. Plan de puesta en producción

### Fase 1. Revisión técnica y de licencias

Se han revisado las licencias de los componentes utilizados y las condiciones de uso de los modelos. La mayoría de los componentes identificados utilizan licencias abiertas, aunque cada dependencia debe conservar sus avisos y condiciones aplicables.

El código del proyecto se distribuye bajo Apache 2.0. Entre los modelos empleados se encuentran Faster Whisper, los modelos de diarización de pyannote y el modelo de estimación de edad y género de audeering.

Se ha identificado que el modelo `audeering/wav2vec2-large-robust-24-ft-age-gender` está sujeto a CC BY-NC-SA 4.0, que establece restricciones para usos comerciales y condiciones de atribución y distribución aplicables. Dado que EQUITalk se plantea como un proyecto gratuito y no comercial, se prevé mantener este modelo respetando sus condiciones. Si en el futuro se modifica la finalidad del proyecto o se introduce una explotación comercial, será necesario revisar esta decisión (por ejemplo, migrando hacia modelos con licencias abiertas tipo CC0 como Common Voice).

La licencia del código no sustituye ni modifica las condiciones de las licencias de los modelos. También se deben respetar los términos de acceso de los repositorios de modelos que los requieran.

Por otra parte, se revisarán las condiciones contractuales y de privacidad de las APIs de inferencia antes de habilitarlas. Las licencias de los modelos no garantizan por sí mismas que un proveedor externo no conserve los datos enviados.

### Fase 2. Validación de las métricas y del rendimiento

Antes de una publicación estable se comprobará que los resultados sean suficientemente consistentes para el uso previsto. Se evaluarán, entre otros aspectos:

- La calidad de la transcripción y la identificación de los turnos de habla.
- La detección de solapamientos e interrupciones.
- La precisión de las métricas de participación y sus posibles errores de cálculo.
- La influencia de los acentos, el ruido, la calidad de la grabación y las características vocales en los resultados.
- La precisión en la detección de dinámicas conversacionales (apertura, cierre y monólogos).

Cuando sea posible, se compararán los resultados con transcripciones o anotaciones de referencia para medir el rendimiento de los modelos. Estas pruebas deberán utilizar datos adecuados para ese fin y contar con las autorizaciones necesarias cuando incluyan voces de personas identificables.

Los resultados se presentarán como estimaciones. No deberán utilizarse como prueba concluyente de discriminación, como evaluación automática del rendimiento laboral ni para adoptar decisiones individuales sin una revisión humana adecuada.

### Fase 3. Prueba piloto controlada

Se realizará una prueba piloto con un conjunto limitado de usuarios y grabaciones cuyo uso sea legítimo. El piloto servirá para evaluar la usabilidad, los tiempos de procesamiento, la calidad de las métricas y los posibles problemas de privacidad.

Durante esta fase se comprobará que la herramienta funcione de forma fluida en entornos locales y que los usuarios comprendan perfectamente las limitaciones de las estimaciones automáticas.

### Fase 4. Protección de datos, privacidad y seguridad

#### 4.1. Procesamiento local y previsión de APIs futuras

El funcionamiento ordinario de EQUITalk está diseñado para ejecutarse íntegramente en local, permitiendo que el audio permanezca en el dispositivo del usuario.

Si en futuras versiones se habilita una alternativa remota para equipos sin GPU, la aplicación deberá mostrar un aviso claro antes de cualquier envío que indique:
- Que el audio saldrá del dispositivo y será procesado por un proveedor externo en la UE.
- Qué datos se enviarán y qué garantías contractuales ofrece el proveedor sobre conservación, eliminación y prohibición de uso para entrenamiento.
- Qué alternativas existen para continuar con el procesamiento local o cancelar la operación.

#### 4.2. Minimización y eliminación de datos

EQUITalk no necesita almacenar permanentemente los audios originales para calcular sus métricas. Por ello, se procesan de forma temporal aplicando las siguientes medidas:
- Eliminar los archivos temporales y las copias locales creadas durante el procesamiento cuando dejen de ser necesarios.
- Evitar guardar audios o transcripciones completas en registros técnicos.
- Permitir que el usuario gestione y elimine los archivos y resultados generados.
- Ofrecer la exportación de métricas agregadas sin datos nominales innecesarios.

#### 4.3. Términos y condiciones de uso (*Terms of Service*)

La interfaz de la aplicación incorporará un mecanismo obligatorio de **aceptación de términos y condiciones de uso** antes de permitir al usuario procesar cualquier archivo de audio. Esto refuerza la concienciación legal sobre el acto de grabar, la privacidad de los interlocutores y la responsabilidad individual en el uso de las grabaciones.

#### 4.4. Información y responsabilidades del usuario

EQUITalk incluirá avisos legales claros en la interfaz, recordando que el usuario debe asegurarse de que la grabación y su análisis sean lícitos y respeten los derechos de las personas afectadas. En particular, se señalará que:
- La participación en una conversación no ampara automáticamente la difusión o el análisis posterior sin consentimiento en determinados ámbitos profesionales u organizativos.
- Las grabaciones y transcripciones contienen datos personales sensibles que deben gestionarse con suma cautela.

## 5. Medidas éticas y perspectiva de género

La función de estimación de género requiere especial precaución. El sistema estima una categoría a partir de características acústicas, pero la voz no permite determinar de forma fiable la identidad de género de una persona. La estimación puede fallar y no debe tratarse como una declaración de identidad.

Por ello, se adoptarán las siguientes medidas:

- Presentar las etiquetas como estimaciones automáticas y no como hechos ciertos.
- Explicar las limitaciones del modelo y la posibilidad de sesgos asociados a voces, acentos, condiciones de grabación y características individuales.
- Permitir, cuando sea viable, corregir las etiquetas o excluir voces del análisis.
- Evitar la identificación nominal de los hablantes cuando no sea necesaria.
- No utilizar las métricas como único fundamento para sanciones, evaluaciones laborales o conclusiones individuales.
- Priorizar los indicadores agregados y el análisis de patrones de interacción frente a la atribución de culpa a personas concretas.

La voz es un dato personal cuando permite identificar directa o indirectamente a una persona. No toda grabación de voz constituye automáticamente un dato biométrico de categoría especial: esta consideración depende, entre otras cuestiones, de si se procesa con la finalidad de identificar de manera única a una persona. En cualquier caso, las grabaciones y las inferencias asociadas deben tratarse con cautela y con garantías adecuadas.

## 6. Criterios para autorizar el despliegue

La publicación de una versión estable quedará condicionada a que se hayan comprobado los siguientes puntos:

- [ ] Las licencias de los modelos y dependencias están documentadas y se respetan sus condiciones.
- [ ] Las métricas se han probado y se explican sus limitaciones y sesgos.
- [ ] El procesamiento local funciona correctamente sin depender de servicios externos.
- [ ] La interfaz incorpora la aceptación obligatoria de términos y condiciones de uso.
- [ ] La aplicación elimina adecuadamente los archivos temporales tras el análisis.
- [ ] Las credenciales están protegidas y no se publican en el repositorio.
- [ ] La interfaz incluye avisos visibles sobre privacidad, uso responsable y la naturaleza orientativa de las estimaciones de género.

## 7. Conclusión

EQUITalk es un proyecto viable como herramienta abierta y gratuita para analizar dinámicas conversacionales, siempre que se mantenga un enfoque de privacidad desde el diseño y se utilicen los resultados de forma responsable.

El objetivo es ofrecer una herramienta útil para reflexionar sobre la equidad en las conversaciones sin convertir estimaciones probabilísticas en hechos sobre las personas ni comprometer su privacidad. La publicación estable dependerá de verificar tanto el funcionamiento técnico como las garantías de seguridad, las licencias y las obligaciones legales aplicables.